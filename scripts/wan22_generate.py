#!/usr/bin/env python3
"""Wan2.2 T2V 文生视频实验 — 红衣少女骑白马朝镜头跑来最后妩媚一笑，多段拼接约2分钟。

范式参照 scripts/ltx_generate.py：diffusers + MPS，多段 prompt 生成 + ffmpeg concat。
冒烟模式先跑 1-2 段验证 pipeline；全量模式 24 段凑约 2 分钟。
"""
import argparse
import os
import sys
import time
import json
import logging
import subprocess
import glob
from pathlib import Path

MODEL_DIR = "/Users/dahai/.cache/wan22-5b"
OUTPUT_DIR = "/tmp/wan22_horse"

SEGMENT_PROMPTS = [
    ("01_wide_gallop", "Wide cinematic shot: a beautiful young Chinese woman in a flowing red dress riding a large white horse, galloping toward the camera across a vast grassland at golden hour, dust kicking up, long lens compression, 4k"),
    ("02_medium_run", "Medium shot: beautiful Chinese girl in red dress on a white horse galloping toward camera, red dress flowing in the wind, long black hair blowing, sunset backlight, slow motion, cinematic"),
    ("03_side_gallop", "Side tracking shot: young woman in red riding white horse galloping across meadow, red dress billowing, horse mane flying, dreamy cinematic quality, golden hour"),
    ("04_close_hair", "Close-up: beautiful Chinese girl in red dress on galloping white horse, wind blowing her long hair across face, warm sunset light, shallow depth of field, cinematic"),
    ("05_close_smile", "Extreme close-up: beautiful Chinese girl in red dress on white horse, looking directly at camera with a charming seductive smile, eyes sparkling, wind blowing hair, golden hour bokeh, cinematic"),
    ("06_wide_smile", "Wide shot: red dress girl on white horse slowing to a trot, looking back at camera with a charming smile, golden grassland, sun flare, cinematic"),
]


def setup_logging(outdir):
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("wan22")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(outdir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_pipeline(model_dir, dtype_str, logger):
    import torch
    from diffusers import WanPipeline

    if not os.path.isdir(model_dir) or not os.path.exists(os.path.join(model_dir, "model_index.json")):
        logger.error("模型未就绪: %s (缺 model_index.json，下载是否完成?)", model_dir)
        sys.exit(1)

    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    logger.info("加载 WanPipeline: %s dtype=%s", model_dir, dtype_str)
    t0 = time.time()
    pipe = WanPipeline.from_pretrained(model_dir, torch_dtype=dtype)
    pipe.to("mps")
    logger.info("pipeline 加载完成 %.0fs", time.time() - t0)
    return pipe, torch


def gen_segment(pipe, torch, prompt, seg_name, seg_idx, outdir, width, height,
                num_frames, steps, guidance, fps, logger):
    seg_dir = os.path.join(outdir, f"seg_{seg_idx:02d}_{seg_name}")
    clip_path = os.path.join(seg_dir, "clip.mp4")
    if os.path.exists(clip_path):
        logger.info("[%02d] %s 已存在，跳过: %s", seg_idx, seg_name, clip_path)
        return clip_path

    os.makedirs(seg_dir, exist_ok=True)
    logger.info("[%02d] %s 生成开始: %dx%d %d帧 %d步 g=%.1f",
                seg_idx, seg_name, width, height, num_frames, steps, guidance)
    logger.info("[%02d] prompt: %s", seg_idx, prompt[:90])
    t0 = time.time()

    generator = torch.Generator(device="cpu").manual_seed(42 + seg_idx)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, cartoon, anime",
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
    frames = result.frames[0]
    gen_secs = time.time() - t0
    logger.info("[%02d] 生成完成 %.0fs (%d帧, %.1fs 视频)",
                seg_idx, gen_secs, len(frames), len(frames) / fps)

    import numpy as np
    from PIL import Image
    for i, frame in enumerate(frames):
        if isinstance(frame, Image.Image):
            img = frame
        else:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            img = Image.fromarray(arr)
        img.save(os.path.join(seg_dir, f"frame_{i:05d}.png"))

    r = subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(seg_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", clip_path,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("[%02d] ffmpeg 编码失败: %s", seg_idx, r.stderr[-500:])
        return None
    logger.info("[%02d] 编码完成: %s", seg_idx, clip_path)
    return clip_path


def concat_segments(clips, outdir, fps, final_name, logger):
    concat_file = os.path.join(outdir, "concat.txt")
    with open(concat_file, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    final = os.path.join(outdir, final_name)
    logger.info("拼接 %d 段 -> %s", len(clips), final)
    r = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file, "-c", "copy", "-movflags", "+faststart", final,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("拼接失败: %s", r.stderr[-500:])
        return None
    rp = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "csv=p=0", final], capture_output=True, text=True)
    dur = float(rp.stdout.strip()) if rp.stdout.strip() else 0
    size = os.path.getsize(final) / 1024 / 1024
    logger.info("最终成片: %s  时长 %.0fs (%.1fmin)  %.0fMB", final, dur, dur / 60, size)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：只跑 1 段")
    ap.add_argument("--segments", type=int, default=0, help="生成段数（0=按 prompt 列表）")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    args = ap.parse_args()

    logger = setup_logging(args.outdir)
    logger.info("=== Wan2.2 T2V 实验启动 ===")
    logger.info("参数: %s", vars(args))

    if args.smoke:
        plan = SEGMENT_PROMPTS[:1]
        final_name = "smoke.mp4"
    else:
        n = args.segments if args.segments > 0 else len(SEGMENT_PROMPTS)
        plan = []
        for i in range(n):
            name, prompt = SEGMENT_PROMPTS[i % len(SEGMENT_PROMPTS)]
            plan.append((f"{name}_{i:02d}", prompt))
        final_name = "final.mp4"
    logger.info("计划 %d 段，单段 %.1fs，目标 %.1fmin",
                len(plan), args.num_frames / args.fps, len(plan) * args.num_frames / args.fps / 60)

    pipe, torch = load_pipeline(args.model_dir, args.dtype, logger)

    clips = []
    for idx, (name, prompt) in enumerate(plan):
        clip = gen_segment(pipe, torch, prompt, name, idx, args.outdir,
                           args.width, args.height, args.num_frames, args.steps,
                           args.guidance, args.fps, logger)
        if clip:
            clips.append(clip)
        else:
            logger.error("段 %d 失败，终止", idx)
            break

    if len(clips) >= 1:
        concat_segments(clips, args.outdir, args.fps, final_name, logger)
    logger.info("=== 完成 ===")


if __name__ == "__main__":
    main()
