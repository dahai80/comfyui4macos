import logging

from ..fusion_client import FusionMLXClient, FusionMLXError, list_models_safe

logger = logging.getLogger("custom_nodes4macos.flux_image")


def _build_prompt(visual_prompt: str, global_style: str) -> str:
    visual_prompt = (visual_prompt or "").strip()
    if not visual_prompt:
        raise ValueError("visual_prompt 不能为空")
    style = (global_style or "").strip()
    if style:
        return f"{visual_prompt}, {style}"
    return visual_prompt


def _bytes_to_image_tensor(img_bytes: bytes):
    import torch
    from PIL import Image
    import numpy as np
    import io

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class FusionMLXFluxImage:
    @classmethod
    def INPUT_TYPES(cls):
        models = list_models_safe()
        return {
            "required": {
                "visual_prompt": ("STRING", {"multiline": True, "default": ""}),
                "global_style": ("STRING", {"multiline": True, "default": "Chinese ink-wash dark fantasy, cinematic, 8k"}),
                "model": (models, {"default": "(auto)"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50, "step": 1}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1}),
            },
            "optional": {
                "base_url": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "FusionMLX/Horror"

    def generate(
        self,
        visual_prompt: str,
        global_style: str,
        model: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        base_url: str = "",
        api_key: str = "",
    ):
        prompt = _build_prompt(visual_prompt, global_style)
        model_name = None if model == "(auto)" else model
        seed_arg = None if not seed else int(seed)
        logger.info(
            "generate prompt_len=%d model=%s size=%dx%d steps=%d guidance=%.1f seed=%s",
            len(prompt), model_name or "auto", width, height, steps, guidance, seed_arg,
        )
        with FusionMLXClient(base_url=base_url or None, api_key=api_key or None) as client:
            if not client.health():
                raise FusionMLXError(f"fusion-mlx 不可达: {client.base_url}")
            images = client.generate_image(
                prompt=prompt,
                model=model_name,
                width=width,
                height=height,
                steps=steps,
                seed=seed_arg,
                guidance=guidance,
                n=1,
                response_format="b64_json",
            )
        if not images:
            raise FusionMLXError("generate_image 返回空结果")
        tensor = _bytes_to_image_tensor(images[0])
        logger.info("generate done tensor_shape=%s", tuple(tensor.shape))
        return (tensor,)
