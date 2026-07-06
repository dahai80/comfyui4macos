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
            description="audio_script → WAV (fusion-mlx HTTP)",
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

        use_fish_s2 = bool(ref_audio and voice_clone_model == "fish-audio-s2-pro")

        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)

        with model_manager.acquire("tts") as handle:
            # D1: Build synthesis plan — deduplicate identical scripts
            # to avoid redundant model inference
            script_to_scenes: dict[str, list[int]] = {}
            scene_scripts: list[tuple[int, str]] = []
            for i, scene in enumerate(ctx.scenes):
                scene_id = scene.get("scene_id", i + 1)
                if ctx.has_artifact_on_disk(scene_id, "audio"):
                    continue
                script = scene.get("audio_script", "")
                if not script or not script.strip():
                    continue
                script_key = script.strip()
                if script_key not in script_to_scenes:
                    script_to_scenes[script_key] = []
                script_to_scenes[script_key].append((i, scene_id, scene))
                scene_scripts.append((i, scene_id, script_key))

            # D1: Synthesize each unique script once, then copy for duplicates
            synthesized_cache: dict[str, str] = {}
            for i, scene_id, script_key in scene_scripts:
                if script_key in synthesized_cache:
                    cached_path = synthesized_cache[script_key]
                    out_path = ctx.artifact_path(scene_id, "audio")
                    import shutil
                    shutil.copy2(cached_path, out_path)
                    ctx.set_artifact(scene_id, "audio", out_path)
                    logger.info(
                        "tts_synthesize scene %d: duplicated from cache (script hash=%s)",
                        scene_id, script_key[:40],
                    )
                    self._set_audio_duration(scene, out_path)
                    ctx.update_progress("tts_synthesize", i + 1, len(ctx.scenes))
                    continue

                scene = ctx.scenes[i]
                scene_chars = scene.get("characters", [])
                scene_instructions = self._get_scene_instructions(
                    instructions, scene_chars, char_lookup,
                )
                out_path = ctx.artifact_path(scene_id, "audio")

                model_name = _FISH_S2_MODEL_ID if use_fish_s2 else handle.model_name
                try:
                    self._synthesize(
                        handle, model_name, scene["audio_script"], voice, scene_instructions,
                        speed, out_path,
                        ref_audio=ref_audio, ref_text=ref_text,
                    )
                except Exception as exc:
                    if not use_fish_s2:
                        raise
                    logger.warning(
                        "[tts_synthesize] Fish S2 HTTP failed (%s), falling back to Qwen3-TTS ICL",
                        exc,
                    )
                    self._synthesize(
                        handle, handle.model_name, scene["audio_script"], voice, scene_instructions,
                        speed, out_path,
                        ref_audio=ref_audio, ref_text=ref_text,
                    )
                ctx.set_artifact(scene_id, "audio", out_path)
                synthesized_cache[script_key] = out_path

                self._set_audio_duration(scene, out_path)

                ctx.update_progress("tts_synthesize", i + 1, len(ctx.scenes))
                if ctx.should_checkpoint_scene(i + 1):
                    checkpoint.save(ctx)
                    logger.info("scene-level checkpoint saved at scene %d", scene_id)

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
                continue
            gender = str(c.get("gender", "")).lower().strip()
            if gender in ("female", "女", "女性", "f"):
                voices.append(f"{name}：女声，温柔细腻")
            elif gender in ("male", "男", "男性", "m"):
                voices.append(f"{name}：男声，低沉稳重")
        if not voices:
            return base_instructions
        return f"{base_instructions}；角色配音：{'；'.join(voices)}"

    @staticmethod
    def _synthesize(
        handle,
        model_name: str,
        text: str,
        voice: str,
        instructions: str,
        speed: float,
        out_path: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> None:
        TTSSynthesizeStage._synthesize_http(
            handle, model_name, text, voice, instructions, speed, out_path,
            ref_audio=ref_audio, ref_text=ref_text,
        )

    @staticmethod
    def _synthesize_http(
        handle,
        model_name: str,
        text: str,
        voice: str,
        instructions: str,
        speed: float,
        out_path: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> None:
        logger.info("tts_synthesize HTTP model=%s text_len=%d", model_name, len(text))
        audio_bytes = handle.client.synthesize_speech(
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

    @staticmethod
    def _set_audio_duration(scene: dict, audio_path: str) -> None:
        try:
            from ...ffmpeg_util import probeDuration
            dur = probeDuration(audio_path)
            if dur > 0:
                scene["duration_seconds"] = dur
        except Exception:
            pass
