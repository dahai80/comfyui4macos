from ..engine import register_stage
from .ken_burns import KenBurnsStage
from .multi_pose import MultiPoseStage
from .assemble import AssembleStage
from .subtitle import SubtitleStage
from .sfx import SFXStage
from .prompt_expand import PromptExpandStage
from .image_generate import ImageGenerateStage
from .tts_synthesize import TTSSynthesizeStage
from .story_ingest import StoryIngestStage
from .digital_human_render import DigitalHumanRenderStage
from .avatar_create import AvatarCreateStage
from .avatar_animate import AvatarAnimateStage
from .voice_clone import VoiceCloneStage
from .series_orchestrate import SeriesOrchestratorStage
from .publish import PublishStage

register_stage(KenBurnsStage)
register_stage(MultiPoseStage)
register_stage(AssembleStage)
register_stage(SFXStage)
register_stage(SubtitleStage)
register_stage(PromptExpandStage)
register_stage(ImageGenerateStage)
register_stage(TTSSynthesizeStage)
register_stage(StoryIngestStage)
register_stage(DigitalHumanRenderStage)
register_stage(AvatarCreateStage)
register_stage(AvatarAnimateStage)
register_stage(VoiceCloneStage)
register_stage(SeriesOrchestratorStage)
register_stage(PublishStage)
