from __future__ import annotations

import logging
import os
import re
import time

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

    @staticmethod
    def _unload_other_models() -> None:
        import httpx
        from ...fusion_client import FusionMLXClient
        client = FusionMLXClient()
        if not client.health():
            logger.warning("tts_synthesize: fusion-mlx 不可达，跳过卸载")
            return
        targets = [
            os.environ.get("FUSION_LLM_MODEL", "Qwen3.5-9B-4bit"),
            os.environ.get("FUSION_FLUX_MODEL", "Flux-1.lite-8B-MLX-Q4"),
            os.environ.get("FUSION_SMALL_LLM_MODEL", "Qwen3-0.6B-4bit"),
        ]
        headers = {"Authorization": f"Bearer {client.api_key}"} if client.api_key else {}
        base = client.base_url.rstrip("/")
        for mid in targets:
            if not mid:
                continue
            try:
                r = httpx.post(f"{base}/admin/api/models/{mid}/unload", headers=headers, timeout=30.0)
                if r.status_code == 200:
                    logger.info("tts_synthesize: 卸载 %s 释放显存", mid)
                elif r.status_code == 400:
                    logger.info("tts_synthesize: %s 未加载，无需卸载", mid)
                else:
                    logger.warning("tts_synthesize: 卸载 %s 返回 %d %s", mid, r.status_code, r.text[:80])
            except Exception as exc:
                logger.warning("tts_synthesize: 卸载 %s 失败(%s)", mid, str(exc)[:80])

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        self._unload_other_models()

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
    def _chunk_text(text: str, max_chars: int) -> list[str]:
        import re
        s = text.strip()
        if len(s) <= max_chars:
            return [s]
        parts = re.split(r'(?<=[。！？!?；;\n])', s)
        parts = [p for p in parts if p.strip()]
        chunks: list[str] = []
        cur = ""
        for p in parts:
            if len(p) > max_chars:
                if cur:
                    chunks.append(cur)
                    cur = ""
                subs = re.split(r'(?<=[，,、：:])', p)
                sc = ""
                for sub in subs:
                    if len(sc) + len(sub) <= max_chars:
                        sc += sub
                    else:
                        if sc:
                            chunks.append(sc)
                        if len(sub) <= max_chars:
                            sc = sub
                        else:
                            for k in range(0, len(sub), max_chars):
                                chunks.append(sub[k:k + max_chars])
                            sc = ""
                if sc:
                    chunks.append(sc)
            elif len(cur) + len(p) <= max_chars:
                cur += p
            else:
                if cur:
                    chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        return chunks

    @staticmethod
    def _sanitize_tts_text(text: str) -> str:
        s = text.replace("《", "").replace("》", "")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _bisect_text(text: str) -> tuple[str, str]:
        n = len(text)
        if n <= 1:
            return text, ""
        marks = [m.end() for m in re.finditer(r"[。！？!?；;\n，,、：:]", text)]
        if not marks:
            return text[: n // 2], text[n // 2 :]
        mid = n // 2
        best = min(marks, key=lambda p: abs(p - mid))
        if best <= 1 or best >= n - 1:
            return text[: n // 2], text[n // 2 :]
        return text[:best], text[best:]

    @staticmethod
    def _make_silence(path: str, seconds: float) -> None:
        from ...ffmpeg_util import run_ffmpeg
        run_ffmpeg(
            ["-f", "lavfi", "-i", f"anullsrc=channel_layout=mono:sample_rate=24000",
             "-t", f"{seconds:.2f}", "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", path],
            label="tts_silence",
        )

    @staticmethod
    def _synthesize_chunk_resilient(
        client,
        model_name: str,
        text: str,
        voice: str,
        instructions: str,
        speed: float,
        tmp_dir: str,
        tag: str,
        ref_audio: str | None,
        ref_text: str | None,
        depth: int = 0,
    ) -> list[str]:
        MAX_DEPTH = 4
        FLOOR = 8
        tmp_path = os.path.join(tmp_dir, f".tts_chunk_{tag}.wav")
        try:
            cb = client.synthesize_speech(
                text=text,
                model=model_name,
                voice=voice or None,
                instructions=instructions or None,
                speed=speed,
                response_format="wav",
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            if not cb:
                raise RuntimeError("synthesize_speech returned empty")
            with open(tmp_path, "wb") as f:
                f.write(cb)
            logger.info("tts_synthesize chunk %s len=%d saved", tag, len(text))
            return [tmp_path]
        except Exception as exc:
            if depth >= MAX_DEPTH or len(text) <= FLOOR:
                logger.warning(
                    "tts_synthesize chunk %s SKIP pathological len=%d text=%r exc=%s",
                    tag, len(text), text[:40], str(exc)[:80],
                )
                sil = os.path.join(tmp_dir, f".tts_silence_{tag}.wav")
                TTSSynthesizeStage._make_silence(sil, 0.6)
                return [sil]
            a, b = TTSSynthesizeStage._bisect_text(text)
            logger.warning(
                "tts_synthesize chunk %s 500/timeout, bisect -> (%d, %d) exc=%s",
                tag, len(a), len(b), str(exc)[:80],
            )
            left = TTSSynthesizeStage._synthesize_chunk_resilient(
                client, model_name, a, voice, instructions, speed, tmp_dir,
                f"{tag}L", ref_audio, ref_text, depth + 1,
            )
            right = TTSSynthesizeStage._synthesize_chunk_resilient(
                client, model_name, b, voice, instructions, speed, tmp_dir,
                f"{tag}R", ref_audio, ref_text, depth + 1,
            )
            return left + right

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
        from ...fusion_client import FusionMLXClient
        text = TTSSynthesizeStage._sanitize_tts_text(text)
        max_chars = int(os.environ.get("TTS_CHUNK_CHARS", "100"))
        chunks = TTSSynthesizeStage._chunk_text(text, max_chars)
        logger.info(
            "tts_synthesize HTTP model=%s text_len=%d chunks=%d",
            model_name, len(text), len(chunks),
        )
        tmp_dir = os.path.dirname(out_path) or "."
        client = FusionMLXClient(timeout=42.0, retries=0)
        chunk_paths: list[str] = []
        try:
            for idx, chunk in enumerate(chunks):
                parts = TTSSynthesizeStage._synthesize_chunk_resilient(
                    client, model_name, chunk, voice, instructions, speed,
                    tmp_dir, str(idx), ref_audio, ref_text,
                )
                chunk_paths.extend(parts)
                time.sleep(0.3)
        finally:
            try:
                client.close()
            except Exception:
                pass
        if len(chunk_paths) == 1:
            import shutil
            shutil.move(chunk_paths[0], out_path)
            logger.info("tts_synthesize HTTP saved: %s (%d bytes)", out_path, os.path.getsize(out_path))
            return
        list_path = os.path.join(tmp_dir, ".tts_concat_list.txt")
        with open(list_path, "w") as f:
            for p in chunk_paths:
                f.write(f"file '{os.path.basename(p)}'\n")
        from ...ffmpeg_util import run_ffmpeg
        run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", list_path,
             "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", out_path],
            label="tts_concat",
        )
        for p in chunk_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.remove(list_path)
        except OSError:
            pass
        logger.info("tts_synthesize HTTP chunked saved: %s (%d bytes)", out_path, os.path.getsize(out_path))

    @staticmethod
    def _set_audio_duration(scene: dict, audio_path: str) -> None:
        try:
            from ...ffmpeg_util import probeDuration
            dur = probeDuration(audio_path)
            if dur > 0:
                scene["duration_seconds"] = dur
        except Exception:
            pass
