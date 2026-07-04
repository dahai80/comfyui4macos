from __future__ import annotations

import logging
import os

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.tts_synthesize")

_FISH_S2_MODEL_ID = "mlx-community/fish-audio-s2-pro"


class TTSSynthesizeStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="tts_synthesize",
            description="audio_script → WAV (mlx_audio native)",
            model_requirements=["tts"],
            memory_estimate_gb=2.9,
            input_kinds=["scenes"],
            output_kinds=["audio"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        voice = ctx.config.get("tts_voice", "")
        instructions = ctx.config.get("tts_instructions", "低沉、压抑、略带颤抖的旁白语气；语速偏慢，停顿处留白以制造悬念")
        speed = ctx.config.get("tts_speed", 1.0)
        character_registry = ctx.config.get("character_registry", [])
        char_lookup = {c["name"]: c for c in character_registry if "name" in c}

        ref_audio = ctx.config.get("ref_audio")
        ref_text = ctx.config.get("ref_text")
        voice_clone_model = ctx.config.get("voice_clone_model", "")

        use_fish_s2 = (
            ref_audio
            and voice_clone_model == "fish-audio-s2-pro"
        )

        fish_s2_model = None
        if use_fish_s2:
            try:
                fish_s2_model = self._load_fish_s2_model()
                logger.info("[tts_synthesize] loaded Fish S2 Pro for voice cloning")
            except Exception as exc:
                logger.warning(
                    "[tts_synthesize] Fish S2 Pro load failed (%s), falling back to Qwen3-TTS ICL",
                    exc,
                )
                use_fish_s2 = False
                fish_s2_model = None

        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)

        with model_manager.acquire("tts") as handle:
            tts_model = handle.model
            for i, scene in enumerate(ctx.scenes):
                scene_id = scene.get("scene_id", i + 1)
                if ctx.has_artifact_on_disk(scene_id, "audio"):
                    logger.info("tts_synthesize scene %d skipped (exists)", scene_id)
                    continue

                audio_script = scene.get("audio_script", "")
                if not audio_script or not audio_script.strip():
                    logger.warning("tts_synthesize scene %d: no audio_script, skip", scene_id)
                    continue

                scene_chars = scene.get("characters", [])
                scene_instructions = self._get_scene_instructions(
                    instructions, scene_chars, char_lookup,
                )

                out_path = ctx.artifact_path(scene_id, "audio")

                if use_fish_s2 and fish_s2_model is not None:
                    self._synthesize_fish_s2(
                        fish_s2_model, audio_script, ref_audio, ref_text,
                        scene_instructions, out_path,
                    )
                else:
                    self._synthesize(
                        tts_model, audio_script, voice, scene_instructions, speed, out_path,
                        ref_audio=ref_audio, ref_text=ref_text,
                    )
                ctx.set_artifact(scene_id, "audio", out_path)

                try:
                    from ...ffmpeg_util import probe_duration
                    dur = probe_duration(out_path)
                    if dur > 0:
                        scene["duration_seconds"] = dur
                        logger.info("tts_synthesize scene %d audio duration=%.2fs", scene_id, dur)
                except Exception:
                    pass

                ctx.update_progress("tts_synthesize", i + 1, len(ctx.scenes))

                if ctx.should_checkpoint_scene(i + 1):
                    checkpoint.save(ctx)
                    logger.info("scene-level checkpoint saved at scene %d", scene_id)

        if fish_s2_model is not None:
            del fish_s2_model
            import gc
            gc.collect()
            logger.info("[tts_synthesize] Fish S2 model released")

    @staticmethod
    def _get_scene_instructions(base_instructions: str, scene_chars: list, char_lookup: dict) -> str:
        if not scene_chars or not char_lookup:
            return base_instructions
        voices = []
        for name in scene_chars:
            c = char_lookup.get(name)
            if not c:
                continue
            if c.get("voice"):
                voices.append(f"{name}：{c['voice']}")
            elif c.get("gender", "").lower() in ("female", "女", "女性", "f"):
                voices.append(f"{name}：女声，温柔细腻")
        if not voices:
            return base_instructions
        return f"{base_instructions}；角色配音：{'；'.join(voices)}"

    @staticmethod
    def _load_fish_s2_model():
        from mlx_audio.tts.utils import load_model
        model = load_model(_FISH_S2_MODEL_ID)
        return model

    @staticmethod
    def _synthesize_fish_s2(
        model,
        text: str,
        ref_audio: str,
        ref_text: str,
        instructions: str,
        out_path: str,
    ) -> None:
        import mlx.core as mx
        import numpy as np

        logger.info(
            "[tts_synthesize] Fish S2 text_len=%d ref_audio=%s ref_text_len=%d",
            len(text), ref_audio, len(ref_text),
        )

        gen_kwargs = {
            "ref_audio": ref_audio,
            "verbose": False,
        }
        if ref_text:
            gen_kwargs["ref_text"] = ref_text
        if instructions:
            gen_kwargs["instruct"] = instructions

        all_audio = []
        for result in model.generate(text, **gen_kwargs):
            audio = result.audio
            if isinstance(audio, mx.array):
                audio = np.array(audio)
            if isinstance(audio, np.ndarray):
                all_audio.append(audio)

        if not all_audio:
            raise RuntimeError("[tts_synthesize] Fish S2 returned no audio")

        full_audio = np.concatenate(all_audio)
        sr = model.sample_rate if hasattr(model, "sample_rate") else 44100
        try:
            import soundfile as sf
            sf.write(out_path, full_audio, sr)
        except ImportError:
            import wave
            arr = full_audio
            if np.issubdtype(arr.dtype, np.floating):
                arr = (arr * 32767).clip(-32767, 32767).astype("int16")
            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(arr.tobytes())

        logger.info(
            "[tts_synthesize] Fish S2 saved: %s (%d bytes) dur=%.1fs",
            out_path, os.path.getsize(out_path), len(full_audio) / sr,
        )

    @staticmethod
    def _synthesize(
        model,
        text: str,
        voice: str,
        instructions: str,
        speed: float,
        out_path: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> None:
        try:
            TTSSynthesizeStage._synthesize_mlx(
                model, text, voice, instructions, speed, out_path,
                ref_audio=ref_audio, ref_text=ref_text,
            )
        except ImportError:
            logger.warning("mlx_audio not available, falling back to HTTP")
            TTSSynthesizeStage._synthesize_http(
                text, voice, instructions, speed, out_path,
                ref_audio=ref_audio, ref_text=ref_text,
            )

    @staticmethod
    def _synthesize_mlx(
        model,
        text: str,
        voice: str,
        instructions: str,
        speed: float,
        out_path: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> None:
        import mlx.core as mx
        import numpy as np

        logger.info("tts_synthesize MLX text_len=%d speed=%.2f voice=%s", len(text), speed, voice or "(default)")
        all_audio = []
        gen_kwargs = {"lang_code": "chinese", "verbose": False}
        if voice:
            gen_kwargs["voice"] = voice
        if instructions:
            gen_kwargs["instructions"] = instructions
        if ref_audio:
            gen_kwargs["ref_audio"] = ref_audio
            logger.info("tts_synthesize MLX using ref_audio for voice cloning")
        if ref_text:
            gen_kwargs["ref_text"] = ref_text

        for result in model.generate(text, **gen_kwargs):
            audio = result.audio
            if isinstance(audio, mx.array):
                audio = np.array(audio)
            if isinstance(audio, np.ndarray):
                all_audio.append(audio)

        if not all_audio:
            raise RuntimeError("tts_synthesize: model.generate returned no audio")

        full_audio = np.concatenate(all_audio)
        sr = model.sample_rate if hasattr(model, "sample_rate") else 24000
        try:
            import soundfile as sf
            sf.write(out_path, full_audio, sr)
        except ImportError:
            import wave
            arr = full_audio
            if np.issubdtype(arr.dtype, np.floating):
                arr = (arr * 32767).clip(-32767, 32767).astype("int16")
            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(arr.tobytes())

        logger.info("tts_synthesize MLX saved: %s (%d bytes) dur=%.1fs",
                     out_path, os.path.getsize(out_path), len(full_audio) / sr)

    @staticmethod
    def _synthesize_http(
        text: str,
        voice: str,
        instructions: str,
        speed: float,
        out_path: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> None:
        from ...fusion_client import FusionMLXClient

        logger.info("tts_synthesize HTTP text_len=%d", len(text))
        with FusionMLXClient() as client:
            if not client.health():
                raise RuntimeError("fusion-mlx unreachable (HTTP fallback)")
            model_name = os.environ.get("FUSION_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
            audio_bytes = client.synthesize_speech(
                text=text,
                model=model_name,
                voice=voice or None,
                instructions=instructions or None,
                speed=speed,
                response_format="wav",
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        if not audio_bytes:
            raise RuntimeError("synthesize_speech returned empty")

        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        logger.info("tts_synthesize HTTP saved: %s (%d bytes)", out_path, os.path.getsize(out_path))
