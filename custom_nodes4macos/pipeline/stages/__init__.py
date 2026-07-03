from ..engine import register_stage
from .ken_burns import KenBurnsStage
from .assemble import AssembleStage
from .prompt_expand import PromptExpandStage
from .image_generate import ImageGenerateStage
from .tts_synthesize import TTSSynthesizeStage
from .story_ingest import StoryIngestStage
from .digital_human_render import DigitalHumanRenderStage

register_stage(KenBurnsStage)
register_stage(AssembleStage)
register_stage(PromptExpandStage)
register_stage(ImageGenerateStage)
register_stage(TTSSynthesizeStage)
register_stage(StoryIngestStage)
register_stage(DigitalHumanRenderStage)
