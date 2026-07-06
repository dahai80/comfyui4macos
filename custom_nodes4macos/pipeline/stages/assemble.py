from __future__ import annotations

import logging
import os

from ..stage import Stage, StageInfo
from ... import ffmpeg_util

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.assemble")

_TRANSITIONS = ["none", "fade", "crossfade", "dissolve", "wipeleft", "wiperight",
                "wipeup", "wipedown", "fadeblack", "fadewhite", "circleopen",
                "circleclose", "horzopen", "horzclose", "radial", "horror"]
_BGM_VOLUME = 0.3
_DUCK_THRESHOLD = 0.04
_DUCK_RATIO = 6
_DUCK_ATTACK_MS = 10
_DUCK_RELEASE_MS = 400
_XFADE_DUR = 0.5
_XFADE_MAP = {
    "crossfade": "fade",
    "dissolve": "dissolve",
    "wipeleft": "wipeleft",
    "wiperight": "wiperight",
    "wipeup": "wipeup",
    "wipedown": "wipedown",
    "fadeblack": "fadeblack",
    "fadewhite": "fadewhite",
    "circleopen": "circleopen",
    "circleclose": "circleclose",
    "horzopen": "horzopen",
    "horzclose": "horzclose",
    "radial": "radial",
    "horror": "fadeblack",
}


class AssembleStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="assemble",
            description="多段 clip → 最终 mp4 (concat + 可选 BGM + 可选淡入淡出)",
            model_requirements=[],
            memory_estimate_gb=0.0,
            input_kinds=["clip"],
            output_kinds=["final"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        width = ctx.config.get("assemble_width", ctx.config.get("ken_burns_width", 1080))
        height = ctx.config.get("assemble_height", ctx.config.get("ken_burns_height", 1920))
        fps = ctx.config.get("assemble_fps", ctx.config.get("ken_burns_fps", 30))
        transition = ctx.config.get("assemble_transition", "none")
        bgm_path = ctx.config.get("assemble_bgm_path", "")
        if not bgm_path:
            bgm_path = self._resolve_bgm_by_mood(ctx)
        duck_bgm = bool(ctx.config.get("assemble_duck_bgm", False))

        clips = self._collect_clips(ctx)
        if not clips:
            raise RuntimeError("assemble: no clips available")

        out_path = ctx.artifact_path(0, "final")

        self._render_final(
            clips, width, height, fps, transition, bgm_path, duck_bgm, out_path,
        )
        ctx.set_artifact(0, "final", out_path)

        friendly = self._friendly_output_path(ctx)
        if friendly and friendly != out_path:
            try:
                import shutil
                if os.path.exists(friendly):
                    os.remove(friendly)
                shutil.copy2(out_path, friendly)
                logger.info("assemble friendly copy: %s → %s", out_path, friendly)
            except Exception as exc:
                logger.warning("failed to create friendly output: %s", exc)

        ctx.update_progress("assemble", 1, 1)

    @staticmethod
    def _friendly_output_path(ctx) -> str | None:
        story_title = ctx.config.get("story_title", "")
        if not story_title and ctx.scenes:
            story_title = ctx.scenes[0].get("story_title", "")
        if not story_title:
            return None
        episode_title = ""
        if ctx.scenes:
            ep_title = ctx.scenes[0].get("episode_title", "")
            if ep_title:
                episode_title = f"_{ep_title}"
        safe_title = "".join(
            c for c in story_title if c.isalnum() or c in "_.-－—"
        ).strip()
        if not safe_title:
            return None
        safe_ep = "".join(
            c for c in episode_title if c.isalnum() or c in "_.-－—"
        ).strip("_-")
        filename = f"{safe_title}{safe_ep}.mp4" if safe_ep else f"{safe_title}.mp4"
        return os.path.join(ctx.job_dir, filename)

    @staticmethod
    def _resolve_bgm_by_mood(ctx) -> str:
        mood_map = ctx.config.get("bgm_mood_map", {})
        if not mood_map:
            return ""
        mood = ctx.config.get("bgm_mood", "")
        if not mood:
            mood = ctx.config.get("style_preset", "")
        if not mood and ctx.scenes:
            mood = ctx.scenes[0].get("bgm_mood", ctx.scenes[0].get("style_preset", ""))
        if not mood:
            return ""
        candidate = mood_map.get(mood, "")
        if candidate and os.path.exists(candidate):
            logger.info("assemble bgm by mood=%s → %s", mood, candidate)
            return candidate
        if candidate:
            logger.warning("assemble bgm_mood_map[%s]=%s missing, ignore", mood, candidate)
        return ""

    @staticmethod
    def _collect_clips(ctx) -> list[str]:
        clips = []
        for i, scene in enumerate(ctx.scenes):
            scene_id = scene.get("scene_id", i + 1)
            clip_path = ctx.get_artifact(scene_id, "clip")
            if clip_path and os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                clips.append(clip_path)
            else:
                logger.warning("assemble scene %d: clip missing, skip", scene_id)
        return clips

    @staticmethod
    def _render_final(
        clips: list[str],
        width: int,
        height: int,
        fps: int,
        transition: str,
        bgm_path: str,
        duck_bgm: bool,
        out_path: str,
    ) -> None:
        n = len(clips)

        xfade_mode = _XFADE_MAP.get(transition)
        if xfade_mode and n >= 2:
            AssembleStage._render_xfade(
                clips, width, height, fps, xfade_mode, bgm_path, duck_bgm, out_path,
            )
            return

        audio_flags = [ffmpeg_util.probe_has_audio(p) for p in clips]
        all_audio = all(audio_flags)
        any_audio = any(audio_flags)
        use_clip_audio = all_audio
        if any_audio and not all_audio:
            logger.warning("assemble clips audio inconsistent, dropping clip audio")
            use_clip_audio = False

        bgm_ok = (
            bgm_path
            and os.path.exists(bgm_path)
            and ffmpeg_util.probe_has_audio(bgm_path)
        )
        if bgm_path and not bgm_ok:
            logger.warning("assemble BGM unusable: %s", bgm_path)

        filters = []
        if use_clip_audio:
            seg_labels = []
            for i in range(n):
                v = f"v{i}"
                a = f"a{i}"
                filters.append(
                    f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                    f"fps={fps},setsar=1,format=yuv420p[{v}]"
                )
                filters.append(f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo[{a}]")
                seg_labels.append(f"[{v}][{a}]")
            concat_in = "".join(seg_labels)
            filters.append(f"{concat_in}concat=n={n}:v=1:a=1[vcat][acat]")
        else:
            v_labels = []
            for i in range(n):
                v = f"v{i}"
                v_labels.append(f"[{v}]")
                filters.append(
                    f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                    f"fps={fps},setsar=1,format=yuv420p[{v}]"
                )
            concat_in = "".join(v_labels)
            filters.append(f"{concat_in}concat=n={n}:v=1:a=0[vcat]")

        video_label = "[vcat]"
        if transition == "fade":
            total = sum(ffmpeg_util.probe_duration(p) for p in clips)
            fade_out_start = max(0.0, total - 0.4)
            filters.append(
                f"[vcat]fade=t=in:st=0:d=0.4,fade=t=out:st={fade_out_start:.3f}:d=0.4[vfade]"
            )
            video_label = "[vfade]"

        bgm_index = n
        if use_clip_audio and bgm_ok:
            if duck_bgm:
                filters.append(f"[acat]asplit=2[narr][key]")
                filters.append(
                    f"[{bgm_index}:a]volume={_BGM_VOLUME},aresample=44100,"
                    f"aformat=channel_layouts=stereo[bgm]"
                )
                filters.append(
                    f"[bgm][key]sidechaincompress="
                    f"threshold={_DUCK_THRESHOLD}:ratio={_DUCK_RATIO}:"
                    f"attack={_DUCK_ATTACK_MS}:release={_DUCK_RELEASE_MS}:makeup=1[ducked]"
                )
                filters.append(
                    "[narr][ducked]amix=inputs=2:duration=first:normalize=0[aout]"
                )
            else:
                filters.append(
                    f"[{bgm_index}:a]volume={_BGM_VOLUME},aresample=44100,"
                    f"aformat=channel_layouts=stereo[bgm]"
                )
                filters.append("[acat][bgm]amix=inputs=2:duration=first:dropout_transition=0[aout]")
            audio_map = "[aout]"
        elif use_clip_audio:
            audio_map = "[acat]"
        elif bgm_ok:
            filters.append(
                f"[{bgm_index}:a]aresample=44100,aformat=channel_layouts=stereo[aout]"
            )
            audio_map = "[aout]"
        else:
            audio_map = None

        filter_complex = ";".join(filters)
        args = []
        for p in clips:
            args += ["-i", p]
        if bgm_ok:
            args += ["-i", bgm_path]
        args += ["-filter_complex", filter_complex]
        args += ["-map", video_label]
        if audio_map:
            args += ["-map", audio_map]
        args += ffmpeg_util.video_encoder_args()
        args += ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]
        if bgm_ok or use_clip_audio:
            args += ["-shortest"]
        args.append(out_path)

        logger.info(
            "assemble n=%d audio=clip:%s bgm:%s transition=%s",
            n, use_clip_audio, bgm_ok, transition,
        )
        ffmpeg_util.run_ffmpeg(args, timeout=900, label=f"assemble n={n}")

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"assemble output empty: {out_path}")

        dur = ffmpeg_util.probe_duration(out_path)
        logger.info("assemble done dur=%.2fs size=%d", dur, os.path.getsize(out_path))

    @staticmethod
    def _render_xfade(
        clips: list[str],
        width: int,
        height: int,
        fps: int,
        xfade_mode: str,
        bgm_path: str,
        duck_bgm: bool,
        out_path: str,
    ) -> None:
        n = len(clips)
        durations = [ffmpeg_util.probe_duration(p) for p in clips]
        min_d = min(durations) if durations else _XFADE_DUR
        D = max(0.1, min(_XFADE_DUR, min_d * 0.5))

        audio_flags = [ffmpeg_util.probe_has_audio(p) for p in clips]
        all_audio = all(audio_flags)
        any_audio = any(audio_flags)
        use_clip_audio = all_audio
        if any_audio and not all_audio:
            logger.warning("assemble xfade clips audio inconsistent, dropping clip audio")
            use_clip_audio = False

        bgm_ok = (
            bgm_path
            and os.path.exists(bgm_path)
            and ffmpeg_util.probe_has_audio(bgm_path)
        )
        if bgm_path and not bgm_ok:
            logger.warning("assemble xfade BGM unusable: %s", bgm_path)

        filters: list[str] = []
        for i in range(n):
            filters.append(
                f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"fps={fps},setsar=1,format=yuv420p[v{i}]"
            )

        prev = "v0"
        acc = durations[0]
        for k in range(1, n):
            offset = max(0.0, acc - D)
            out_label = f"vx{k}" if k < n - 1 else "vfinal"
            filters.append(
                f"[{prev}][v{k}]xfade=transition={xfade_mode}:"
                f"duration={D:.3f}:offset={offset:.3f}[{out_label}]"
            )
            acc = acc + durations[k] - D
            prev = out_label
        video_label = f"[{prev}]"

        audio_map: str | None = None
        if use_clip_audio:
            for i in range(n):
                filters.append(
                    f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo[a{i}]"
                )
            aprev = "a0"
            for k in range(1, n):
                out_label = f"ax{k}" if k < n - 1 else "anarr"
                filters.append(f"[{aprev}][a{k}]acrossfade=d={D:.3f}[{out_label}]")
                aprev = out_label
            narr_label = f"[{aprev}]"

            bgm_index = n
            if bgm_ok:
                if duck_bgm:
                    filters.append(f"{narr_label}asplit=2[narr][key]")
                    filters.append(
                        f"[{bgm_index}:a]volume={_BGM_VOLUME},aresample=44100,"
                        f"aformat=channel_layouts=stereo[bgm]"
                    )
                    filters.append(
                        f"[bgm][key]sidechaincompress="
                        f"threshold={_DUCK_THRESHOLD}:ratio={_DUCK_RATIO}:"
                        f"attack={_DUCK_ATTACK_MS}:release={_DUCK_RELEASE_MS}:makeup=1[ducked]"
                    )
                    filters.append(
                        "[narr][ducked]amix=inputs=2:duration=first:normalize=0[aout]"
                    )
                else:
                    filters.append(
                        f"[{bgm_index}:a]volume={_BGM_VOLUME},aresample=44100,"
                        f"aformat=channel_layouts=stereo[bgm]"
                    )
                    filters.append(
                        f"{narr_label}[bgm]amix=inputs=2:duration=first:dropout_transition=0[aout]"
                    )
                audio_map = "[aout]"
            else:
                audio_map = narr_label
        elif bgm_ok:
            bgm_index = n
            filters.append(
                f"[{bgm_index}:a]aresample=44100,aformat=channel_layouts=stereo[aout]"
            )
            audio_map = "[aout]"

        filter_complex = ";".join(filters)
        args = []
        for p in clips:
            args += ["-i", p]
        if bgm_ok:
            args += ["-i", bgm_path]
        args += ["-filter_complex", filter_complex, "-map", video_label]
        if audio_map:
            args += ["-map", audio_map]
        args += ffmpeg_util.video_encoder_args()
        args += ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]
        if bgm_ok or use_clip_audio:
            args += ["-shortest"]
        args.append(out_path)

        logger.info(
            "assemble xfade n=%d mode=%s D=%.2f audio=clip:%s bgm:%s duck:%s",
            n, xfade_mode, D, use_clip_audio, bgm_ok, duck_bgm,
        )
        ffmpeg_util.run_ffmpeg(args, timeout=900, label=f"assemble xfade n={n}")

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"assemble xfade output empty: {out_path}")

        dur = ffmpeg_util.probe_duration(out_path)
        logger.info("assemble xfade done dur=%.2fs size=%d", dur, os.path.getsize(out_path))
