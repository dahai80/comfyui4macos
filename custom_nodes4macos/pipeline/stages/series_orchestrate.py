from __future__ import annotations

import copy
import logging
import os
import shutil

from ..context import PipelineContext
from ..stage import Stage, StageInfo
from .prompt_expand import PromptExpandStage
from .image_generate import ImageGenerateStage
from .tts_synthesize import TTSSynthesizeStage
from .wan_video import WanVideoStage
from .ken_burns import KenBurnsStage
from .assemble import AssembleStage
from .sfx import SFXStage
from .subtitle import SubtitleStage
from .avatar_create import AvatarCreateStage
from .voice_clone import VoiceCloneStage
from .avatar_animate import AvatarAnimateStage

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.series_orchestrate")

_CN_DIGITS = "零一二三四五六七八九"


def _cn_numeral(n: int) -> str:
    if n <= 0:
        return str(n)
    if n < 10:
        return _CN_DIGITS[n]
    if n == 10:
        return "十"
    if n < 20:
        return "十" + (_CN_DIGITS[n - 10] if n - 10 > 0 else "")
    if n < 100:
        tens = n // 10
        ones = n % 10
        s = _CN_DIGITS[tens] + "十"
        if ones > 0:
            s += _CN_DIGITS[ones]
        return s
    return str(n)


class SeriesOrchestratorStage(Stage):

    _PER_EPISODE_STAGES = (
        PromptExpandStage,
        ImageGenerateStage,
        TTSSynthesizeStage,
        WanVideoStage,
        KenBurnsStage,
        AssembleStage,
        SFXStage,
        SubtitleStage,
    )

    _DIGITAL_HUMAN_STAGES = (
        AvatarCreateStage,
        VoiceCloneStage,
        PromptExpandStage,
        TTSSynthesizeStage,
        AvatarAnimateStage,
        AssembleStage,
    )

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="series_orchestrate",
            description="分集编排：逐集跑子流水线，跨集共享角色注册表，产出独立分集 mp4",
            model_requirements=["llm", "flux", "tts"],
            memory_estimate_gb=6.0,
            input_kinds=["episodes"],
            output_kinds=["final"],
        )

    @staticmethod
    def _is_digital_human_mode(ctx) -> bool:
        avatar_pkg = ctx.config.get("avatar_package", "")
        avatar_ref = ctx.config.get("avatar_reference", "")
        if avatar_pkg and os.path.isdir(avatar_pkg):
            return True
        if avatar_ref and os.path.isfile(avatar_ref):
            return True
        return False

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        episodes = ctx.config.get("episodes", [])
        if not episodes:
            logger.warning("series_orchestrate: no episodes, nothing to do")
            return

        shared_registry = ctx.config.setdefault("character_registry", [])
        completed = list(ctx.config.get("_completed_episodes", []))
        episode_finals = list(ctx.config.get("_episode_finals", []))
        story_title = ctx.config.get("story_title") or "series"
        total = len(episodes)

        digital_human_mode = self._is_digital_human_mode(ctx)
        stage_classes = self._DIGITAL_HUMAN_STAGES if digital_human_mode else self._PER_EPISODE_STAGES
        per_episode_stages = [cls() for cls in stage_classes]
        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)

        logger.info(
            "series_orchestrate: %d episodes, %d already completed, title=%s, mode=%s",
            total, len(completed), story_title,
            "digital_human" if digital_human_mode else "image",
        )

        for ep_idx, episode in enumerate(episodes):
            ep_num = ep_idx + 1
            if ep_num in completed:
                logger.info("series_orchestrate: episode %d skipped (completed)", ep_num)
                continue

            ep_title = episode.get("title", f"第{ep_num}集")
            ep_dir = os.path.join(ctx.job_dir, f"episode_{ep_num:02d}")
            os.makedirs(ep_dir, exist_ok=True)

            sub_config = copy.deepcopy(ctx.config)
            sub_config["episodes"] = [episode]
            sub_config.pop("stages", None)
            sub_config.pop("_completed_episodes", None)
            sub_config.pop("_episode_finals", None)
            sub_config["character_registry"] = shared_registry
            sub_config["episode_title"] = ep_title

            sub_ctx = PipelineContext(
                job_id=f"{ctx.job_id}_ep{ep_num}",
                job_dir=ep_dir,
                config=sub_config,
            )

            logger.info(
                "series_orchestrate: episode %d/%d '%s' starting → %s",
                ep_num, total, ep_title, ep_dir,
            )
            for stage in per_episode_stages:
                stage_info = stage.info()
                logger.info(
                    "series_orchestrate ep%d: stage %s starting",
                    ep_num, stage_info.name,
                )
                stage.process(sub_ctx, model_manager)

            ep_final = sub_ctx.get_artifact(0, "final")
            if not ep_final or not os.path.exists(ep_final) or os.path.getsize(ep_final) == 0:
                logger.error(
                    "series_orchestrate: episode %d produced no final video (path=%s)",
                    ep_num, ep_final,
                )
                raise RuntimeError(
                    f"series_orchestrate: episode {ep_num} final video missing or empty: {ep_final}"
                )

            dest = os.path.join(ctx.job_dir, f"{story_title}_第{_cn_numeral(ep_num)}集.mp4")
            shutil.copy2(ep_final, dest)
            episode_finals.append(dest)
            completed.append(ep_num)
            ctx.config["_completed_episodes"] = completed
            ctx.config["_episode_finals"] = episode_finals
            ctx.config["character_registry"] = shared_registry

            if ctx.config.get("cleanup_episode_intermediates", True):
                self._cleanup_episode_intermediates(ep_dir)

            checkpoint.save(ctx)
            ctx.update_progress("series_orchestrate", ep_num, total)
            logger.info(
                "series_orchestrate: episode %d/%d done → %s (%.1f KB)",
                ep_num, total, dest, os.path.getsize(dest) / 1024.0,
            )

        ctx.config["_episode_finals"] = episode_finals
        if episode_finals:
            ctx.set_artifact(0, "final", episode_finals[-1])
        logger.info(
            "series_orchestrate: all episodes done, %d finals: %s",
            len(episode_finals), episode_finals,
        )

    _CLEANUP_SUFFIXES = ("_image.png", "_audio.wav", "_clip.mp4")

    @staticmethod
    def _cleanup_episode_intermediates(ep_dir: str) -> None:
        removed = 0
        freed_kb = 0.0
        try:
            names = os.listdir(ep_dir)
        except OSError as e:
            logger.warning("series_orchestrate cleanup: cannot list %s: %s", ep_dir, e)
            return
        for name in names:
            if not name.startswith("scene_"):
                continue
            if not any(name.endswith(suf) for suf in SeriesOrchestratorStage._CLEANUP_SUFFIXES):
                continue
            path = os.path.join(ep_dir, name)
            try:
                size = os.path.getsize(path)
                os.remove(path)
                removed += 1
                freed_kb += size / 1024.0
            except OSError as e:
                logger.warning("series_orchestrate cleanup: skip %s: %s", path, e)
        if removed:
            logger.info(
                "series_orchestrate cleanup: removed %d intermediates (%.1f KB) from %s",
                removed, freed_kb, ep_dir,
            )
