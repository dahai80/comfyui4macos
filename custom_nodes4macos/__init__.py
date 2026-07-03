import logging
import os

logger = logging.getLogger("custom_nodes4macos")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [custom_nodes4macos] %(levelname)s %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(os.environ.get("CUSTOM_NODES4MACOS_LOG_LEVEL", "INFO"))
logger.info("loading custom_nodes4macos")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

try:
    from .nodes.prompt_expand import FusionMLXPromptExpand

    NODE_CLASS_MAPPINGS["FusionMLXPromptExpand"] = FusionMLXPromptExpand
    NODE_DISPLAY_NAME_MAPPINGS["FusionMLXPromptExpand"] = "FusionMLX Prompt Expand (Horror Director)"
    logger.info("registered node FusionMLXPromptExpand")
except Exception as exc:
    logger.exception("failed to register prompt_expand node: %s", exc)

try:
    from .nodes.flux_image import FusionMLXFluxImage

    NODE_CLASS_MAPPINGS["FusionMLXFluxImage"] = FusionMLXFluxImage
    NODE_DISPLAY_NAME_MAPPINGS["FusionMLXFluxImage"] = "FusionMLX Flux Image (Horror Visual)"
    logger.info("registered node FusionMLXFluxImage")
except Exception as exc:
    logger.exception("failed to register flux_image node: %s", exc)

try:
    from .nodes.horror_tts import FusionMLXHorrorTTS

    NODE_CLASS_MAPPINGS["FusionMLXHorrorTTS"] = FusionMLXHorrorTTS
    NODE_DISPLAY_NAME_MAPPINGS["FusionMLXHorrorTTS"] = "FusionMLX Horror TTS (Eerie Narration)"
    logger.info("registered node FusionMLXHorrorTTS")
except Exception as exc:
    logger.exception("failed to register horror_tts node: %s", exc)

try:
    from .nodes.ken_burns import FusionMLXKenBurns

    NODE_CLASS_MAPPINGS["FusionMLXKenBurns"] = FusionMLXKenBurns
    NODE_DISPLAY_NAME_MAPPINGS["FusionMLXKenBurns"] = "FusionMLX Ken Burns (Still→9:16)"
    logger.info("registered node FusionMLXKenBurns")
except Exception as exc:
    logger.exception("failed to register ken_burns node: %s", exc)

try:
    from .nodes.assemble import FusionMLXAssemble

    NODE_CLASS_MAPPINGS["FusionMLXAssemble"] = FusionMLXAssemble
    NODE_DISPLAY_NAME_MAPPINGS["FusionMLXAssemble"] = "FusionMLX Assemble (Clips→Drama)"
    logger.info("registered node FusionMLXAssemble")
except Exception as exc:
    logger.exception("failed to register assemble node: %s", exc)

try:
    from .nodes.dream_factory import FusionMLXDreamFactory

    NODE_CLASS_MAPPINGS["FusionMLXDreamFactory"] = FusionMLXDreamFactory
    NODE_DISPLAY_NAME_MAPPINGS["FusionMLXDreamFactory"] = "FusionMLX Dream Factory (梦工厂)"
    logger.info("registered node FusionMLXDreamFactory")
except Exception as exc:
    logger.exception("failed to register dream_factory node: %s", exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
