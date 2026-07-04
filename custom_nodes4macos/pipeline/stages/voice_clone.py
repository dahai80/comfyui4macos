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

        from mlx_audio.utils import load_audio
        audio = load_audio(ref_audio_path, sample_rate=24000)
        duration = audio.shape[0] / 24000
        if duration < 3.0:
            raise ValueError(f"voice ref audio too short: {duration:.1f}s, need >=3s")

        profile_dir = os.path.join(ctx.job_dir, "_voice_profile")
        os.makedirs(profile_dir, exist_ok=True)

        ref_wav_path = os.path.join(profile_dir, "voice_ref.wav")
        audio_np = np.array(audio, dtype=np.float32)
        try:
            import soundfile as sf
            sf.write(ref_wav_path, audio_np, 24000)
        except ImportError:
            import wave
            arr = (audio_np * 32767).clip(-32767, 32767).astype("int16")
            with wave.open(ref_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(arr.tobytes())

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
            from mlx_audio.stt.utils import load_model as load_stt_model
            from mlx_audio.stt.generate import generate_transcription

            stt_model = load_stt_model("mlx-community/whisper-large-v3-turbo")
            result = generate_transcription(
                model=stt_model, audio=audio_path, format="txt",
            )
            text = getattr(result, "text", "").strip()
            del stt_model
            import gc
            gc.collect()
            return text
        except Exception as exc:
            logger.warning("[voice_clone] auto-transcribe failed: %s", exc)
            return ""
