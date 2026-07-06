from __future__ import annotations

import json
import logging
import os

import numpy as np

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.voice_clone")


class VoiceCloneStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="voice_clone",
            description="从参考音频创建声音 profile，供下游 TTS 使用克隆声色",
            model_requirements=[],
            memory_estimate_gb=1.5,
            input_kinds=["audio"],
            output_kinds=["voice_profile"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        ref_audio_path = ctx.config.get("voice_ref_audio", "")
        ref_text_input = ctx.config.get("voice_ref_text", "")
        voice_clone_model = ctx.config.get("voice_clone_model", "fish-audio-s2-pro")

        if not ref_audio_path:
            logger.info("[voice_clone] no voice_ref_audio, skipping")
            return

        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"voice ref audio not found: {ref_audio_path}")

        import soundfile as sf
        audio, sr = sf.read(ref_audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 24000:
            n_out = int(len(audio) * 24000 / sr)
            x_old = np.linspace(0, 1, num=len(audio))
            x_new = np.linspace(0, 1, num=n_out)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
        duration = len(audio) / 24000
        if duration < 3.0:
            raise ValueError(f"voice ref audio too short: {duration:.1f}s, need >=3s")

        profile_dir = os.path.join(ctx.job_dir, "_voice_profile")
        os.makedirs(profile_dir, exist_ok=True)

        ref_wav_path = os.path.join(profile_dir, "voice_ref.wav")
        sf.write(ref_wav_path, audio, 24000)

        ref_text = ref_text_input.strip()
        ref_text_source = "user"
        if not ref_text:
            logger.info("[voice_clone] no ref_text, auto-transcribing with Whisper")
            ref_text = self._auto_transcribe(ref_audio_path)
            ref_text_source = "auto_whisper"
            logger.info("[voice_clone] auto-transcribed: %s", ref_text[:100])

        meta = {
            "source_audio": ref_audio_path,
            "ref_wav_path": ref_wav_path,
            "ref_text": ref_text,
            "ref_text_source": ref_text_source,
            "voice_clone_model": voice_clone_model,
            "duration": round(duration, 2),
            "sample_rate": 24000,
        }
        meta_path = os.path.join(profile_dir, "voice_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        ctx.config["ref_audio"] = ref_wav_path
        ctx.config["ref_text"] = ref_text
        ctx.config["voice_clone_model"] = voice_clone_model
        ctx.artifacts["voice_profile"] = profile_dir

        logger.info(
            "[voice_clone] profile created: %s (%.1fs, model=%s, ref_text=%s)",
            ref_wav_path, duration, voice_clone_model, ref_text_source,
        )

    @staticmethod
    def _auto_transcribe(audio_path: str) -> str:
        try:
            from ...fusion_client import FusionMLXClient
            with FusionMLXClient() as client:
                if not client.health():
                    logger.warning("[voice_clone] fusion-mlx unreachable, skip transcribe")
                    return ""
                text, _ = client.transcribe(audio_path)
            return (text or "").strip()
        except Exception as exc:
            logger.warning("[voice_clone] auto-transcribe failed: %s", exc)
            return ""
