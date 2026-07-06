#!/usr/bin/env python3
"""LTX-Video 生成脚本 — 模型下载完成后直接跑"""
import sys, os, time, subprocess, glob, json
from PIL import Image
import numpy as np

# Model paths
MODELSCOPE_CACHE = os.path.expanduser("~/.cache/modelscope/hub/models")
LTX_MODEL_PATH = os.path.join(MODELSCOPE_CACHE, "Lightricks/LTX-Video")
# Also check HF cache
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--Lightricks--LTX-Video/snapshots")
if os.path.isdir(HF_CACHE):
    commits = sorted(os.listdir(HF_CACHE))
    if commits:
        LTX_MODEL_PATH = os.path.join(HF_CACHE, commits[-1])

OUTPUT_DIR = "/tmp/horse_video"
INPUT_IMAGE = os.path.join(OUTPUT_DIR, "scene_05.png")

def main():
    if not os.path.isdir(LTX_MODEL_PATH):
        print(f"LTX model not found at {LTX_MODEL_PATH}")
        # Try diffusers from_pretrained
        LTX_HF_ID = "Lightricks/LTX-Video"
        print(f"Trying HuggingFace: {LTX_HF_ID}")
    else:
        print(f"Using local model: {LTX_MODEL_PATH}")
        LTX_HF_ID = LTX_MODEL_PATH

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from diffusers import LTXImageToVideoPipeline
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading LTX pipeline...")
    t0 = time.time()
    pipe = LTXImageToVideoPipeline.from_pretrained(
        LTX_HF_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    print(f"Loaded in {time.time()-t0:.0f}s")

    # Load input image
    img = Image.open(INPUT_IMAGE).convert("RGB")
    img = img.resize((512, 512))
    print(f"Input image: {img.size}")

    # Generate multiple segments
    prompts = [
        "A young woman in flowing red dress riding a white horse galloping towards camera, smiling, wind blowing her hair and dress, cinematic golden hour, slow motion",
        "Close-up of beautiful Chinese girl in red dress on white horse, laughing joyfully, horse galloping, sunset backlight, cinematic slow motion",
        "Young woman in red riding white horse, side view galloping through meadow, red dress flowing in wind, dreamy cinematic quality",
    ]

    all_segments = []
    for seg_idx, prompt in enumerate(prompts):
        print(f"\nSegment {seg_idx+1}/{len(prompts)}: {prompt[:50]}...")
        t1 = time.time()
        result = pipe(
            image=img,
            prompt=prompt,
            width=512, height=512,
            num_frames=49,  # ~2 seconds at 24fps
            num_inference_steps=30,
            guidance_scale=4.0,
        )
        frames = result.frames[0]
        seg_dir = os.path.join(OUTPUT_DIR, f"ltx_seg_{seg_idx:02d}")
        os.makedirs(seg_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            if isinstance(frame, Image.Image):
                frame.save(os.path.join(seg_dir, f"frame_{i:05d}.png"))
        # Encode to video
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "24",
            "-i", os.path.join(seg_dir, "frame_%05d.png"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            os.path.join(seg_dir, "clip.mp4"),
        ], capture_output=True)
        all_segments.append(os.path.join(seg_dir, "clip.mp4"))
        print(f"  Done in {time.time()-t1:.0f}s")

    # Concatenate all segments
    print(f"\nConcatenating {len(all_segments)} segments...")
    concat_file = os.path.join(OUTPUT_DIR, "ltx_concat.txt")
    with open(concat_file, "w") as f:
        for seg in all_segments:
            f.write(f"file '{seg}'\n")
    final = os.path.join(OUTPUT_DIR, "ltx_final.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        "-movflags", "+faststart",
        final,
    ], capture_output=True)

    # Check result
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", final], capture_output=True, text=True)
    duration = float(r.stdout.strip()) if r.stdout.strip() else 0
    size = os.path.getsize(final) / 1024 / 1024
    print(f"\n✅ Final video: {final}")
    print(f"   Duration: {duration:.0f}s ({duration/60:.1f}min)")
    print(f"   Size: {size:.0f}MB")

if __name__ == "__main__":
    main()
