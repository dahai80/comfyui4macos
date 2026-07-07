#!/usr/bin/env python3
"""西游记连续剧：每回一集 20-40min，Wan2.2 i2v 驱动角色动作与表情。

用法:
  # 全量 99 回（后台长跑，约 99 × 4-8h）
  EPISODE_COUNT=99 SCENE_COUNT=20 python run_xiyou.py

  # 第一回端到端验证（30min 输出，约 7-8h）
  EPISODE_COUNT=1 SCENE_COUNT=20 python run_xiyou.py

  # 快速冒烟（验证接线，关闭 wan）
  EPISODE_COUNT=1 SCENE_COUNT=2 WAN_ENABLED=0 python run_xiyou.py

  # wan 接线冒烟（低步数，约 15min）
  EPISODE_COUNT=1 SCENE_COUNT=2 WAN_STEPS=4 python run_xiyou.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

os.environ.setdefault("PYTHONPATH", ".")
# 8192-token 思考+JSON 生成在 9B-4bit MPS 上约 4-5min，默认 120s 客户端超时太紧。
os.environ.setdefault("FUSION_MLX_TIMEOUT", "600")
# fusion-mlx admin API（卸载模型）鉴权；本地 key，非敏感。
os.environ.setdefault("FUSION_MLX_API_KEY", "dahai168")
# TTS 分块阈值：fusion-mlx 内部 TTS 超时 ~60s；speed=0.95 + GPU 竞争时 220 字→59.9s（3.7字/s）。
# 160 字/块在最差竞争下约 43s（28% 余量），480 字场景拆 3 块，规避 500。
os.environ.setdefault("TTS_CHUNK_CHARS", "160")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_xiyou")

STORY_FILE = os.path.expanduser("~/Downloads/西游记.txt")

CHARACTER_REGISTRY = [
    {
        "name": "孙悟空",
        "appearance": (
            "Sun Wukong the Monkey King: golden fiery eyes, thunder-beak mouth, "
            "lean muscular monkey-human hybrid, wearing a tiger-skin skirt and "
            "golden armor, wielding the golden staff Ruyi Jingu Bang, "
            "Chinese face, East Asian features"
        ),
        "voice": "男声，尖锐洪亮，桀骜不驯，语速偏快，带猴性顽皮",
    },
    {
        "name": "唐三藏",
        "appearance": (
            "Tang Sanzang the Tang Monk: gentle benevolent face, clean-shaven, "
            "wearing a crimson monk robe with golden kasaya, holding a Buddhist "
            "staff and alms bowl, Chinese face, East Asian features"
        ),
        "voice": "男声，温和沉稳，慈悲庄重，语速缓慢，带梵音韵味",
    },
    {
        "name": "猪八戒",
        "appearance": (
            "Zhu Bajie Pigsy: pig-head with large ears and snout, fat bulky body, "
            "carrying a nine-tooth iron rake, wearing dark monk robe, "
            "Chinese mythological creature"
        ),
        "voice": "男声，粗憨沙哑，贪吃懒散，语速拖沓，带憨笑",
    },
    {
        "name": "沙悟净",
        "appearance": (
            "Sha Wujing Sandy: tall sturdy man with red beard and a necklace of "
            "skulls, calm resolute face, carrying a monk's spade staff, wearing "
            "grey monk robe, Chinese face, East Asian features"
        ),
        "voice": "男声，低沉憨厚，忠诚寡言，语速平稳",
    },
]


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "")
    if val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _prewarm_model(name: str, action, timeout: float = 300.0) -> None:
    """轮询直到模型首次成功响应，规避 fusion-mlx 冷加载 >内部超时 触发 500。

    fusion-mlx 内部对 chat/speech 有 30-60s 超时，冷加载（40-60s）会触发 TimeoutError→500，
    但加载在后台继续。这里反复请求直到首次成功，确保流水线各 stage 不被冷加载拖垮。
    fusion-mlx 多模型常驻不驱逐（已验证 LLM/TTS 同时 warm），可安全预加载全部模型。
    """
    from custom_nodes4macos.fusion_client import FusionMLXClient
    client = FusionMLXClient()
    if not client.health():
        logger.warning("prewarm %s: fusion-mlx 不可达，跳过", name)
        return
    logger.info("prewarm %s（冷加载可能 30-60s）...", name)
    t0 = time.time()
    for attempt in range(1, 13):
        try:
            action(client)
            logger.info("prewarm %s OK attempt=%d %.1fs", name, attempt, time.time() - t0)
            return
        except Exception as exc:
            logger.info("prewarm %s attempt=%d 失败(%s)，15s 后重试", name, attempt, str(exc)[:80])
            time.sleep(15)
    logger.warning("prewarm %s: 12 次重试耗尽(%.1fs)，继续运行", name, time.time() - t0)


def _prewarm() -> None:
    """预加载 LLM/Flux/TTS 三个模型，规避冷加载 500。"""
    llm = os.environ.get("FUSION_LLM_MODEL", "Qwen3.5-9B-4bit")
    tts = os.environ.get("FUSION_TTS_MODEL", "Qwen3-TTS-12Hz-1.7B-Base-8bit")
    flux = os.environ.get("FUSION_FLUX_MODEL", "Flux-1.lite-8B-MLX-Q4")
    _prewarm_model(
        "LLM",
        lambda c: c.chat([{"role": "user", "content": "hi"}], model=llm, max_tokens=5, timeout=300.0),
    )
    if _env_bool("PREWARM_FLUX", True):
        _prewarm_model(
            "Flux",
            lambda c: c.generate_image("a red apple", model=flux, width=256, height=256, steps=2, n=1, timeout=300.0),
        )
    _prewarm_model(
        "TTS",
        lambda c: c.synthesize_speech("测试", model=tts, timeout=300.0),
    )


def main():
    if not os.path.isfile(STORY_FILE):
        logger.error("story file not found: %s", STORY_FILE)
        sys.exit(1)
    logger.info("story file: %s (%d bytes)", STORY_FILE, os.path.getsize(STORY_FILE))

    episode_count = _env_int("EPISODE_COUNT", 99)
    scene_count = _env_int("SCENE_COUNT", 20)
    wan_enabled = _env_bool("WAN_ENABLED", True)
    wan_steps = _env_int("WAN_STEPS", 20)
    keep_intermediates = _env_bool("KEEP_INTERMEDIATES", False)

    _prewarm()

    from custom_nodes4macos.pipeline import PipelineEngine
    engine = PipelineEngine()

    t_total = time.time()
    logger.info(
        "=== 西游记 PIPELINE START === episode_count=%d scene_count=%d "
        "wan_enabled=%s wan_steps=%d",
        episode_count, scene_count, wan_enabled, wan_steps,
    )

    result = engine.run(
        content_type="series",
        story_file=STORY_FILE,
        story_title="西游记",
        episode_count=episode_count,
        scene_count=scene_count,
        style_preset="电影叙事",
        one_episode_per_chapter=True,
        story_ingest_max_chapters=100,
        wan_enabled=wan_enabled,
        wan_frames=41,
        wan_steps=wan_steps,
        wan_seed=1001,
        flux_steps=3,
        flux_guidance=4.0,
        flux_width=832,
        flux_height=1440,
        flux_model=os.environ.get("FUSION_FLUX_MODEL", "Flux-1.lite-8B-MLX-Q4"),
        consistency_check=True,
        checkpoint_every_n_scenes=1,
        ken_burns_workers=3,
        ken_burns_fps=24,
        ken_burns_render_fps=12,
        cleanup_episode_intermediates=not keep_intermediates,
        character_registry=CHARACTER_REGISTRY,
        resume_from=os.environ.get("RESUME_FROM") or None,
    )

    t_elapsed = time.time() - t_total
    logger.info("=== 西游记 PIPELINE COMPLETE ===")
    logger.info("job_id: %s", result.job_id)
    logger.info("job_dir: %s", result.job_dir)
    logger.info("final_video: %s", result.final_video)
    logger.info("total time: %.1fs (%.1fmin)", t_elapsed, t_elapsed / 60)

    cp_path = os.path.join(result.job_dir, "_checkpoint.json")
    if os.path.isfile(cp_path):
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        logger.info("completed episodes: %s", cp.get("_completed_episodes", []))
        logger.info("episode finals: %s", cp.get("_episode_finals", []))


if __name__ == "__main__":
    main()
