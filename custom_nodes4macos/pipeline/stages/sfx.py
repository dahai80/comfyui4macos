from __future__ import annotations

import logging
import os

from ..stage import Stage, StageInfo
from ... import ffmpeg_util
from .assemble import AssembleStage

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.sfx")

_DEFAULT_SCENE_SECONDS = 8.0


class SFXStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="sfx",
            description="按场景 sound_effect 关键词叠加音效到 final 音轨（时间戳对齐）",
            model_requirements=[],
            memory_estimate_gb=0.0,
            input_kinds=["final"],
            output_kinds=["final"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        final_path = ctx.get_artifact(0, "final")
        if not final_path or not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            logger.warning("sfx: final artifact missing, skip sfx stage")
            ctx.update_progress("sfx", 1, 1)
            return

        sfx_map = ctx.config.get("sfx_map", {})
        if not sfx_map:
            logger.info("sfx: no sfx_map configured, skip sfx stage")
            ctx.update_progress("sfx", 1, 1)
            return

        picks = self._collect_sfx(ctx, sfx_map)
        if not picks:
            logger.info("sfx: no sound_effect matched sfx_map, skip sfx stage")
            ctx.update_progress("sfx", 1, 1)
            return

        try:
            self._render_sfx(final_path, picks)
            logger.info("sfx: layered %d effect(s) onto final", len(picks))
        except Exception as exc:
            logger.error("sfx: render failed: %s (final unchanged)", exc)
            ctx.update_progress("sfx", 1, 1)
            return

        self._refresh_friendly(ctx, final_path)
        ctx.update_progress("sfx", 1, 1)

    @staticmethod
    def _collect_sfx(ctx, sfx_map: dict) -> list[tuple[str, float]]:
        picks: list[tuple[str, float]] = []
        cursor = 0.0
        for i, scene in enumerate(ctx.scenes):
            dur = float(scene.get("duration_seconds", scene.get("duration", _DEFAULT_SCENE_SECONDS)))
            text = str(scene.get("sound_effect", ""))
            matched = None
            if text:
                for keyword, path in sfx_map.items():
                    if keyword and keyword in text and path and os.path.exists(path):
                        matched = path
                        break
            if matched:
                picks.append((matched, cursor))
                logger.info("sfx scene=%d t=%.2fs → %s", i + 1, cursor, matched)
            cursor += dur
        return picks

    @staticmethod
    def _render_sfx(final_path: str, picks: list[tuple[str, float]]) -> None:
        has_final_audio = ffmpeg_util.probe_has_audio(final_path)

        filters: list[str] = []
        amix_inputs: list[str] = []
        if has_final_audio:
            filters.append("[0:a]aresample=44100,aformat=channel_layouts=stereo[orig_a]")
            amix_inputs.append("[orig_a]")

        for idx, (sfx_path, start) in enumerate(picks, start=1):
            ms = max(0, int(round(start * 1000)))
            label = f"sfx{idx}"
            filters.append(
                f"[{idx}:a]adelay={ms}|{ms},aresample=44100,"
                f"aformat=channel_layouts=stereo[{label}]"
            )
            amix_inputs.append(f"[{label}]")

        n_inputs = 1 + len(picks)
        amix_in_str = "".join(amix_inputs)
        if has_final_audio:
            filters.append(
                f"{amix_in_str}amix=inputs={len(amix_inputs)}:duration=first:normalize=0[aout]"
            )
            audio_map = "[aout]"
        else:
            filters.append(
                f"{amix_in_str}amix=inputs={len(amix_inputs)}:duration=longest:normalize=0[aout]"
            )
            audio_map = "[aout]"

        filter_complex = ";".join(filters)
        args = ["-i", final_path]
        for sfx_path, _ in picks:
            args += ["-i", sfx_path]
        args += ["-filter_complex", filter_complex, "-map", "0:v", "-map", audio_map]
        args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]

        tmp_path = final_path + ".sfxtmp.mp4"
        args += ["-y", tmp_path]
        ffmpeg_util.run_ffmpeg(args, timeout=600, label="sfx layer")

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError(f"sfx output empty: {tmp_path}")
        os.replace(tmp_path, final_path)

    @staticmethod
    def _refresh_friendly(ctx, final_path: str) -> None:
        friendly = AssembleStage._friendly_output_path(ctx)
        if friendly and friendly != final_path:
            try:
                import shutil
                if os.path.exists(friendly):
                    os.remove(friendly)
                shutil.copy2(final_path, friendly)
                logger.info("sfx friendly copy: %s → %s", final_path, friendly)
            except Exception as exc:
                logger.warning("sfx: friendly copy failed: %s", exc)
