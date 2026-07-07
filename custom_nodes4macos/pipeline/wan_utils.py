from __future__ import annotations

import logging
import os
import sys
import time

logger = logging.getLogger("custom_nodes4macos.pipeline.wan_utils")

REPO = os.path.expanduser("~/digital-man/Wan2.2")
DEFAULT_CKPT = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Wan-AI/Wan2___2-TI2V-5B"
)

_pipe_cache: dict[tuple[str, str], object] = {}


def _ensure_repo() -> None:
    if not os.path.isdir(REPO):
        raise FileNotFoundError(f"Wan2.2 repo not found: {REPO}")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


def load_wan_pipe(checkpoint_dir: str = "", device_id: str = "mps"):
    ckpt = os.path.expanduser(checkpoint_dir or DEFAULT_CKPT)
    key = (ckpt, device_id)
    if key in _pipe_cache:
        logger.info("wan pipe reuse from cache ckpt=%s", ckpt)
        return _pipe_cache[key]

    import torch
    _ensure_repo()
    import wan
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS["ti2v-5B"]
    logger.info("loading wan2.2 TI2V-5B pipeline ckpt=%s device=%s ...", ckpt, device_id)
    t0 = time.time()
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=ckpt,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=False,
    )
    pipe.model.float()
    _install_cpu_vae_decode(pipe, torch)
    logger.info("wan pipeline ready %.1fs", time.time() - t0)
    _pipe_cache[key] = pipe
    return pipe


def _install_cpu_vae_decode(pipe, torch) -> None:
    _orig = pipe.vae.decode
    logger.info("install CPU VAE decode patch (dodges MPS NDArrayMatMul abort)")

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
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
        zs_cpu = [z.to("cpu") for z in zs]
        tdec = time.time()
        out = _orig(zs_cpu)
        logger.info("vae decode cpu %.1fs out_none=%s", time.time() - tdec, out is None)
        pipe.vae.model.to("mps")
        pipe.vae.scale = scale_orig
        if out is None:
            return None
        return [o.to("mps") for o in out]

    pipe.vae.decode = _cpu_vae_decode


def wan_i2v(
    pipe,
    prompt: str,
    img,
    out_path: str,
    frames: int = 41,
    size: tuple = (480, 832),
    max_area: int = 399360,
    steps: int = 20,
    fps: int = 24,
    seed: int = 1001,
) -> str:
    if frames % 4 != 1:
        raise ValueError(f"frames must be 4n+1, got {frames}")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        logger.info("wan i2v skip (exists): %s", out_path)
        return out_path

    from wan.utils.utils import save_video
    logger.info(
        "wan i2v frames=%d size=%s max_area=%d steps=%d fps=%d seed=%d",
        frames, size, max_area, steps, fps, seed,
    )
    t0 = time.time()
    video = pipe.generate(
        prompt,
        img=img,
        size=size,
        max_area=max_area,
        frame_num=frames,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=steps,
        guide_scale=5.0,
        seed=seed,
        offload_model=True,
    )
    logger.info("wan gen done %.1fs tensor=%s", time.time() - t0, tuple(video.shape))
    save_video(video[None], save_file=out_path, fps=fps, nrow=1)
    logger.info(
        "wan saved -> %s (%dKB)",
        out_path, os.path.getsize(out_path) // 1024,
    )
    return out_path
