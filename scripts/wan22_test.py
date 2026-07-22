#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time

REPO = os.path.expanduser("~/digital-man/Wan2.2")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CKPT = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Wan-AI/Wan2___2-TI2V-5B"
)
EXAMPLE_IMG = os.path.join(CKPT, "examples/i2v_input.JPG")

CAT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a "
    "surfboard. The fluffy-furred feline gazes directly at the camera with a "
    "relaxed expression. Blurred beach scenery forms the background featuring "
    "crystal-clear waters, distant green hills, and a blue sky dotted with "
    "white clouds. The cat assumes a naturally relaxed posture, as if savoring "
    "the sea breeze and warm sunlight. A close-up shot highlights the feline's "
    "intricate details and the refreshing atmosphere of the seaside."
)


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Wan2.2 TI2V-5B MPS smoke test")
    p.add_argument("--task", default="ti2v-5B", choices=["ti2v-5B"])
    p.add_argument("--ckpt-dir", default=CKPT)
    p.add_argument("--image", default=EXAMPLE_IMG,
                   help="input image for i2v; 'none' to run t2v")
    p.add_argument("--prompt", default=CAT_PROMPT)
    p.add_argument("--size", default="480*832",
                   help="size key in SIZE_CONFIGS / MAX_AREA_CONFIGS")
    p.add_argument("--max-area", type=int, default=None,
                   help="override max_area (pixels); defaults to MAX_AREA_CONFIGS[size]. "
                        "Use a small value (e.g. 114688) on MPS to bound attention memory.")
    p.add_argument("--frames", type=int, default=17, help="must be 4n+1")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--solver", default="unipc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/tmp/wan22_i2v_test.mp4")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--device", default="mps")
    p.add_argument("--t5-cpu", dest="t5_cpu", action="store_true", default=True)
    p.add_argument("--no-t5-cpu", dest="t5_cpu", action="store_false")
    p.add_argument("--offload", action="store_true", default=False)
    p.add_argument("--convert-dtype", dest="convert_dtype",
                   action="store_true", default=False)
    p.add_argument("--no-convert-dtype", dest="convert_dtype",
                   action="store_false")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    import torch
    from PIL import Image
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS, SIZE_CONFIGS
    import wan
    from wan.utils.utils import save_video

    assert torch.backends.mps.is_available(), "MPS not available"
    assert args.frames % 4 == 1, f"frames must be 4n+1, got {args.frames}"
    assert args.size in MAX_AREA_CONFIGS, f"unknown size key: {args.size}"

    cfg = WAN_CONFIGS[args.task]
    logging.info(
        f"task={args.task} device={args.device} frames={args.frames} "
        f"steps={args.steps} size={args.size} t5_cpu={args.t5_cpu} "
        f"offload={args.offload} convert_dtype={args.convert_dtype}"
    )
    logging.info(f"ckpt_dir={args.ckpt_dir}")
    logging.info(f"prompt={args.prompt[:80]}...")

    img = None
    if args.image and args.image.lower() != "none":
        assert os.path.exists(args.image), f"image not found: {args.image}"
        img = Image.open(args.image).convert("RGB")
        logging.info(f"image={args.image} size={img.size}")

    t0 = time.time()
    logging.info("Creating WanTI2V pipeline ...")
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_dtype,
    )
    logging.info(f"pipeline ready in {time.time()-t0:.1f}s")

    logging.info("casting DiT to float32 for MPS (avoids mixed-dtype matmul); "
                 "T5 stays bf16, context cast to float32 in model.py")
    wan_ti2v.model.float()

    t1 = time.time()
    logging.info("Generating video ...")
    video = wan_ti2v.generate(
        args.prompt,
        img=img,
        size=SIZE_CONFIGS[args.size],
        max_area=args.max_area if args.max_area else MAX_AREA_CONFIGS[args.size],
        frame_num=args.frames,
        shift=args.shift,
        sample_solver=args.solver,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        offload_model=args.offload,
    )
    gen_time = time.time() - t1
    logging.info(
        f"generation done in {gen_time:.1f}s  "
        f"output tensor shape={tuple(video.shape)} dtype={video.dtype}"
    )

    # save individual frames for visual inspection (save_video tiles into a grid)
    try:
        n_frames = video.shape[1]
        for fi in sorted({0, n_frames // 2, n_frames - 1}):
            fr = video[:, fi, :, :].clamp(-1, 1)
            fr = ((fr + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(fr).save(f"/tmp/wan22_frame_{fi:02d}.png")
        logging.info(f"saved inspection frames: 0/{n_frames // 2}/{n_frames - 1}")
    except Exception as e:
        logging.warning(f"frame PNG save failed: {e}")

    logging.info(f"saving to {args.out} fps={args.fps}")
    # save_video expects 5D [B,C,T,H,W]; passing 4D makes unbind(2) split H
    # instead of T, producing 384 frames of 17px height (matches ffprobe
    # nb_frames=384 height=32). Match generate.py: add [None] and nrow=1.
    save_video(video[None], save_file=args.out, fps=args.fps, nrow=1)

    if not os.path.exists(args.out):
        logging.error("save_video produced no file (see 'save_video failed' above)")
        sys.exit(1)
    out_kb = os.path.getsize(args.out) / 1024
    logging.info(f"output: {args.out} ({out_kb:.1f} KB)")
    if out_kb < 5:
        logging.error(f"output file suspiciously small ({out_kb:.1f} KB)")
        sys.exit(1)

    print(f"\nDONE: {args.out}  frames={args.frames} steps={args.steps} "
          f"gen={gen_time:.1f}s size={out_kb:.1f}KB")


if __name__ == "__main__":
    main()
