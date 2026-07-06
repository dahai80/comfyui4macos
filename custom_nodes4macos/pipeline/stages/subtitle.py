from __future__ import annotations

import logging
import os
import re
import subprocess

from ..stage import Stage, StageInfo
from ... import ffmpeg_util
from .assemble import AssembleStage

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.subtitle")

_SENTENCE_ENDS = "。！？!?；;\n"
_SOFT_BREAKS = "，,、：: 　"
_MAX_CHARS = 18
_MIN_CUE_SECONDS = 0.8
_DEFAULT_SCENE_SECONDS = 8.0

_BURN_CAPABLE: bool | None = None


class SubtitleStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="subtitle",
            description="audio_script+duration → SRT；渲染字幕到 final（burn→soft→none 降级）",
            model_requirements=[],
            memory_estimate_gb=0.0,
            input_kinds=["final"],
            output_kinds=["final", "subtitle"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        final_path = ctx.get_artifact(0, "final")
        if not final_path or not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            logger.warning("subtitle: final artifact missing, skip subtitle stage")
            ctx.update_progress("subtitle", 1, 1)
            return

        srt_text = self._build_srt(ctx.scenes)
        srt_path = ctx.artifact_path(0, "subtitle")
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write(srt_text)
        ctx.set_artifact(0, "subtitle", srt_path)
        logger.info("subtitle: srt written (%d cues) → %s", srt_text.count(" --> "), srt_path)

        mode = str(ctx.config.get("subtitle_mode", "auto")).lower().strip()
        rendered = False

        want_burn = mode in ("auto", "burn")
        if want_burn and self._can_burn():
            try:
                self._render_burn(ctx, final_path, srt_path)
                rendered = True
                logger.info("subtitle: burned hardsubs (libass) onto final")
            except Exception as exc:
                logger.warning("subtitle: burn failed (%s), falling back to soft-sub", exc)

        if not rendered and mode in ("auto", "soft"):
            try:
                self._render_soft(final_path, srt_path)
                rendered = True
                logger.info("subtitle: muxed soft subs (mov_text) into final")
            except Exception as exc:
                logger.error("subtitle: soft mux failed: %s", exc)

        if not rendered:
            logger.warning(
                "subtitle: render skipped (mode=%s burn_capable=%s); shipping SRT + clean final",
                mode, self._can_burn(),
            )

        self._refresh_friendly(ctx, final_path)
        ctx.update_progress("subtitle", 1, 1)

    @staticmethod
    def _build_srt(scenes: list[dict]) -> str:
        cues_all: list[tuple[str, float, float]] = []
        offset = 0.0
        for i, scene in enumerate(scenes):
            script = (scene.get("audio_script") or "").strip()
            dur = float(scene.get("duration_seconds") or scene.get("duration") or _DEFAULT_SCENE_SECONDS)
            if dur <= 0:
                dur = _DEFAULT_SCENE_SECONDS
            if script:
                for text, start, end in SubtitleStage._distribute_cues(
                    SubtitleStage._split_into_cues(script), offset, dur,
                ):
                    cues_all.append((text, start, end))
            offset += dur

        lines: list[str] = []
        for idx, (text, start, end) in enumerate(cues_all, 1):
            lines.append(str(idx))
            lines.append(f"{SubtitleStage._format_timestamp(start)} --> {SubtitleStage._format_timestamp(end)}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _split_into_cues(text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        sentence_segs: list[str] = []
        buf = ""
        for ch in text:
            buf += ch
            if ch in _SENTENCE_ENDS:
                if buf.strip():
                    sentence_segs.append(buf.strip())
                buf = ""
        if buf.strip():
            sentence_segs.append(buf.strip())

        cues: list[str] = []
        for seg in sentence_segs:
            fragments: list[str] = []
            piece = ""
            for ch in seg:
                piece += ch
                if ch in _SOFT_BREAKS:
                    fragments.append(piece)
                    piece = ""
            if piece:
                fragments.append(piece)

            cur = ""
            for frag in fragments:
                frag = frag.strip()
                if not frag:
                    continue
                if len(frag) > _MAX_CHARS:
                    if cur:
                        cues.append(cur)
                        cur = ""
                    cues.extend(SubtitleStage._hard_split(frag, _MAX_CHARS))
                    continue
                if cur and len(cur) + len(frag) > _MAX_CHARS:
                    cues.append(cur)
                    cur = frag
                else:
                    cur = cur + frag if cur else frag
            if cur:
                cues.append(cur)
        return [c for c in cues if c]

    @staticmethod
    def _hard_split(text: str, max_chars: int) -> list[str]:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    @staticmethod
    def _distribute_cues(
        cues: list[str], scene_start: float, scene_dur: float,
    ) -> list[tuple[str, float, float]]:
        if not cues:
            return []
        total_chars = sum(len(c) for c in cues)
        if total_chars <= 0:
            return []
        raw = [scene_dur * len(c) / total_chars for c in cues]
        adj = [max(_MIN_CUE_SECONDS, r) for r in raw]
        total_adj = sum(adj)
        scale = scene_dur / total_adj if total_adj > 0 else 1.0
        adj = [a * scale for a in adj]

        result: list[tuple[str, float, float]] = []
        t = scene_start
        for i, c in enumerate(cues):
            start = t
            end = scene_start + scene_dur if i == len(cues) - 1 else t + adj[i]
            result.append((c, start, end))
            t = end
        return result

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        if seconds < 0:
            seconds = 0.0
        total_ms = int(round(seconds * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        m = (total_s // 60) % 60
        h = total_s // 3600
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _can_burn() -> bool:
        global _BURN_CAPABLE
        if _BURN_CAPABLE is not None:
            return _BURN_CAPABLE
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True, text=True, timeout=15,
            )
            _BURN_CAPABLE = re.search(r"(?m)^[\w.]+\s+subtitles\b", proc.stdout) is not None
        except Exception as exc:
            logger.warning("subtitle: ffmpeg filter probe failed: %s", exc)
            _BURN_CAPABLE = False
        logger.info("subtitle: burn capability = %s", _BURN_CAPABLE)
        return _BURN_CAPABLE

    @staticmethod
    def _render_burn(ctx, final_path: str, srt_path: str) -> None:
        font = str(ctx.config.get("subtitle_font", "Arial Unicode"))
        font_size = int(ctx.config.get("subtitle_font_size", 20))
        margin_v = int(ctx.config.get("subtitle_margin_v", 60))
        force_style = (
            f"FontName={font},FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV={margin_v}"
        )
        vf = f"subtitles={srt_path}:force_style='{force_style}'"
        tmp_path = final_path + ".subtmp.mp4"
        args = [
            "-i", final_path,
            "-vf", vf,
        ]
        args += ffmpeg_util.video_encoder_args()
        args += ["-c:a", "copy", "-movflags", "+faststart", tmp_path]
        ffmpeg_util.run_ffmpeg(args, timeout=900, label="subtitle burn")
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError(f"subtitle burn output empty: {tmp_path}")
        os.replace(tmp_path, final_path)

    @staticmethod
    def _render_soft(final_path: str, srt_path: str) -> None:
        tmp_path = final_path + ".subtmp.mp4"
        args = [
            "-i", final_path,
            "-i", srt_path,
            "-c", "copy",
            "-c:s", "mov_text",
            "-movflags", "+faststart",
            tmp_path,
        ]
        ffmpeg_util.run_ffmpeg(args, timeout=600, label="subtitle soft mux")
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError(f"subtitle soft mux output empty: {tmp_path}")
        os.replace(tmp_path, final_path)

    @staticmethod
    def _refresh_friendly(ctx, final_path: str) -> None:
        friendly = AssembleStage._friendly_output_path(ctx)
        if not friendly or friendly == final_path:
            return
        try:
            import shutil
            if os.path.exists(friendly):
                os.remove(friendly)
            shutil.copy2(final_path, friendly)
            logger.info("subtitle: refreshed friendly copy → %s", friendly)
        except Exception as exc:
            logger.warning("subtitle: failed to refresh friendly copy: %s", exc)
