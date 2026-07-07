#!/usr/bin/env python3
import argparse
import base64
import io
import json
import logging
import os
import subprocess
import sys
import time

import requests
from PIL import Image

REPO = os.path.expanduser("~/digital-man/Wan2.2")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FUSION_URL = "http://127.0.0.1:11434"
FLUX_MODEL = "Flux-1.lite-8B-MLX-Q4"
SHOTS_JSON = "/tmp/wan22_shots.json"
OUT_DIR = "/tmp/wan22_drama"
FINAL_MP4 = "/tmp/wan22_drama.mp4"

SIZE = (480, 832)
MAX_AREA = 399360
FRAMES = 81
FPS = 24
STEPS = 20
FLUX_W, FLUX_H = 832, 1408


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def flux_generate(prompt, seed, out_path, steps=8, guidance=4.0):
    if os.path.exists(out_path):
        logging.info(f"skip flux (exists): {out_path}")
        return out_path
    payload = {
        "prompt": prompt,
        "model": FLUX_MODEL,
        "width": FLUX_W,
        "height": FLUX_H,
        "steps": steps,
        "guidance": guidance,
        "n": 1,
        "response_format": "b64_json",
        "seed": seed,
    }
    logging.info(f"flux gen size={FLUX_W}x{FLUX_H} steps={steps} seed={seed}")
    t0 = time.time()
    r = requests.post(f"{FUSION_URL}/v1/images/generate", json=payload, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"flux HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    b64 = data["data"][0]["b64_json"]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img.save(out_path)
    dt = time.time() - t0
    logging.info(
        f"flux done {dt:.1f}s -> {out_path} "
        f"({os.path.getsize(out_path) // 1024}KB) src_size={img.size}"
    )
    return out_path


def wan_i2v(pipe, prompt, img, seed, out_path):
    if os.path.exists(out_path):
        logging.info(f"skip wan i2v (exists): {out_path}")
        return out_path
    from wan.utils.utils import save_video
    logging.info(
        f"wan i2v frames={FRAMES} size={SIZE} max_area={MAX_AREA} "
        f"steps={STEPS} seed={seed} offload=True"
    )
    t0 = time.time()
    video = pipe.generate(
        prompt,
        img=img,
        size=SIZE,
        max_area=MAX_AREA,
        frame_num=FRAMES,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=STEPS,
        guide_scale=5.0,
        seed=seed,
        offload_model=True,
    )
    dt = time.time() - t0
    logging.info(f"wan gen done {dt:.1f}s tensor={tuple(video.shape)}")
    save_video(video[None], save_file=out_path, fps=FPS, nrow=1)
    logging.info(f"wan saved -> {out_path} ({os.path.getsize(out_path) // 1024}KB)")
    return out_path


def ffmpeg_concat(segs, out):
    if os.path.exists(out):
        logging.info(f"skip concat (exists): {out}")
        return out
    listf = os.path.join(OUT_DIR, "concat.txt")
    with open(listf, "w") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    logging.info(f"ffmpeg concat {len(segs)} segs -> {out}")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", listf, "-c", "copy", out,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logging.warning(f"concat copy failed, re-encode: {r.stderr[-300:]}")
        cmd2 = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", listf, "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p", out,
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {r2.stderr[-500:]}")
    return out


def ffprobe_verify(path):
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
            "-of", "default=noprint_wrappers=1", path,
        ],
        capture_output=True, text=True,
    )
    logging.info(f"FFPROBE {path}\n{r.stdout.strip()}")
    return r.stdout


def main():
    global FRAMES, MAX_AREA
    p = argparse.ArgumentParser(description="Wan2.2 3-shot drama: flux first-frame + i2v + concat")
    p.add_argument("--shots", default=SHOTS_JSON)
    p.add_argument(
        "--only", default="all",
        choices=["all", "frames", "video", "shot1", "shot2", "shot3", "concat"],
    )
    p.add_argument("--frames", type=int, default=FRAMES)
    p.add_argument("--max-area", type=int, default=MAX_AREA)
    args = p.parse_args()
    setup_logging()
    os.makedirs(OUT_DIR, exist_ok=True)

    FRAMES = args.frames
    MAX_AREA = args.max_area
    assert FRAMES % 4 == 1, f"frames must be 4n+1, got {FRAMES}"

    shots = json.load(open(args.shots))["shots"]
    for i, s in enumerate(shots, 1):
        s["_frame"] = os.path.join(OUT_DIR, f"shot{i}_frame.png")
        s["_seg"] = os.path.join(OUT_DIR, f"shot{i}.mp4")
    segs = [s["_seg"] for s in shots]

    if args.only in ("all", "frames"):
        logging.info("=== PHASE 1: flux first frames ===")
        for i, s in enumerate(shots, 1):
            flux_generate(s["first_frame_prompt"], s["seed"], s["_frame"])

    if args.only in ("all", "video", "shot1", "shot2", "shot3"):
        logging.info("=== PHASE 2: wan2.2 i2v ===")
        import torch
        import wan
        from wan.configs import WAN_CONFIGS
        cfg = WAN_CONFIGS["ti2v-5B"]
        CKPT = os.path.expanduser(
            "~/.cache/modelscope/hub/models/Wan-AI/Wan2___2-TI2V-5B"
        )
        logging.info("loading wan pipeline ...")
        t0 = time.time()
        pipe = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=CKPT,
            device_id="mps",
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
            convert_model_dtype=False,
        )
        pipe.model.float()
        logging.info(f"pipeline ready {time.time() - t0:.1f}s")
        # MPS NDArray MatMul aborts (SIGABRT) on VAE decode conv3d shape —
        # crash report: MPSNDArrayMatMulA14DeviceBehavior::EncodeArrayMultiply
        # -> abort. DiT sampling matmul is fine on MPS; only VAE decode triggers.
        # Force VAE decode onto CPU to dodge the MPS matmul fast path.
        _orig_vae_decode = pipe.vae.decode
        # encode() (1 frame) is fine on MPS; only 81-frame decode() hits the
        # MPS NDArray MatMul abort. For decode: move model+scale+latent to cpu,
        # run, then restore to mps so the next shot's encode() still works.
        def _cpu_vae_decode(zs):
            if not isinstance(zs, list):
                zs = [zs]
            pipe.vae.model.cpu()
            scale_orig = pipe.vae.scale
            if isinstance(scale_orig, (tuple, list)):
                pipe.vae.scale = tuple(
                    s.to("cpu") if isinstance(s, torch.Tensor) else s
                    for s in scale_orig
                )
            torch.mps.empty_cache()
            zs_cpu = [z.to("cpu") for z in zs]
            tdec = time.time()
            out = _orig_vae_decode(zs_cpu)
            logging.info(
                f"vae decode cpu {time.time() - tdec:.1f}s out_none={out is None}"
            )
            pipe.vae.model.to("mps")
            pipe.vae.scale = scale_orig
            if out is None:
                return None
            return [o.to("mps") for o in out]
        pipe.vae.decode = _cpu_vae_decode
        logging.info("VAE decode -> CPU (MPS matmul abort workaround)")
        for i, s in enumerate(shots, 1):
            if args.only not in ("all", "video", f"shot{i}"):
                continue
            assert os.path.exists(s["_frame"]), f"first frame missing: {s['_frame']}"
            img = Image.open(s["_frame"]).convert("RGB")
            try:
                wan_i2v(pipe, s["i2v_prompt"], img, s["seed"], s["_seg"])
            except Exception as e:
                logging.error(f"shot{i} i2v failed: {e}")
                raise

    if args.only in ("all", "concat"):
        logging.info("=== PHASE 3: ffmpeg concat ===")
        for s in segs:
            assert os.path.exists(s), f"segment missing: {s}"
        ffmpeg_concat(segs, FINAL_MP4)
        ffprobe_verify(FINAL_MP4)
        out_kb = os.path.getsize(FINAL_MP4) / 1024
        print(f"\nDONE: {FINAL_MP4}  shots={len(segs)} size={out_kb:.1f}KB")


if __name__ == "__main__":
    main()
