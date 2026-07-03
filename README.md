# comfyui4macos

MLX-native AI content production pipeline for macOS. One-click short drama, series, digital human, and puppet show — all running locally on Apple Silicon.

## Architecture

**Dream Factory (梦工厂)** — a pluggable pipeline engine inside ComfyUI:

```
PipelineEngine
  ├── Stage ABC (pluggable pipeline steps)
  │     ├── StoryIngestStage     — PDF/EPUB/TXT → chapter split → episode outline
  │     ├── PromptExpandStage    — story seed → structured scene JSON (LLM)
  │     ├── ImageGenerateStage   — visual_prompt → PNG (FluxPipeline MLX)
  │     ├── TTSSynthesizeStage   — audio_script → WAV (mlx_audio)
  │     ├── KenBurnsStage        — PNG + audio → mp4 clip (ffmpeg)
  │     ├── AssembleStage        — clips → final mp4 (ffmpeg concat)
  │     └── DigitalHumanRenderStage — avatar + audio → video (fallback: static composite)
  ├── ModelManager    — sequential model lifecycle (acquire → use → release + mx.clear_cache)
  ├── CheckpointManager — stage + scene-level checkpointing, resume from any point
  └── PipelineContext  — file-based artifact store (PNG/WAV/MP4 on disk)
```

### Content Types (YAML templates, zero Python code to add new ones)

| Type | Template | Duration | Key Feature |
|------|----------|----------|-------------|
| 短剧 Short Drama | `short_drama.yaml` | 1-10min | Static images + Ken Burns + TTS |
| 广告剧 Ad Drama | `ad_drama.yaml` | 1-5min | Brand placement + product showcase |
| 木偶剧 Puppet Show | `puppet_show.yaml` | 5-15min | Character consistency + scene animation |
| 中短视频 Medium Video | `medium_video.yaml` | 30min | Checkpoint/resume + scene-level idempotency |
| 连续剧 Series | `series.yaml` | 30 episodes × 25min | PDF story ingestion → auto-split → 30-episode series |
| 数字人 Digital Human | `digital_human.yaml` | Variable | TTS + digital avatar + lip sync (stub) |
| 数字人直播 Digital Human Live | `digital_human_live.yaml` | Real-time | Placeholder for future livestream |

### MLX Native Models

| Model | Size | Purpose |
|-------|------|---------|
| Qwen3.5-9B-4bit | 5.6G | Prompt expansion + story ingestion |
| Flux-1.lite-8B-MLX-Q4 | 7.0G | Image generation (4-step denoising) |
| Qwen3-TTS-12Hz-1.7B-Base-8bit | 2.9G | Text-to-speech |

**Sequential loading**: Peak GPU memory = 7.0G (Flux only), not 15.5G (all three).

### Performance Optimizations

- **VideoToolbox h264**: Auto-detected hardware encoding on Apple Silicon
- **Parallel Ken Burns**: Configurable worker threads (`ken_burns_workers: 2-3`)
- **Ultrafast preset**: FFmpeg `-preset ultrafast` for clip rendering
- **Auto memory budget**: Detects system RAM via `sysctl hw.memsize`, reserves 40% + 4G
- **Scene-level checkpointing**: Resume mid-stage without re-processing completed scenes
- **Lazy MLX imports**: All `import mlx_*` inside stage methods; no crash without MLX
- **HTTP fallback**: `FusionMLXClient` when MLX not available

## Quick Start

```bash
# Setup
cd comfyui4macos
python3 -m venv .venv && source .venv/bin/activate
pip install torch mlx mlx-lm mlx-audio Pillow numpy scipy pyyaml httpx

# Run tests (all mock, no MLX required)
PYTHONPATH=. python -m pytest custom_nodes4macos/tests/ -v

# MLX live test (requires models in HuggingFace cache)
PYTHONPATH=. python -c "
from custom_nodes4macos.pipeline import PipelineEngine
engine = PipelineEngine()
result = engine.run('short_drama', story_seed='深夜赶路遇白衣女子')
print(result)
"
```

## ComfyUI Integration

Single node `FusionMLXDreamFactory` encapsulates the entire pipeline:

- **Input**: `content_type`, `story_seed`, `story_file`, `episode_count`, `avatar_reference`, config overrides
- **Output**: `video_path`, `scenes_json` (with progress tracking)
- **Resume**: Pass `resume_job_id` to continue from checkpoint

## File Structure

```
custom_nodes4macos/
├── __init__.py                  # Register FusionMLXDreamFactory + 5 legacy nodes
├── fusion_client.py             # HTTP bridge (legacy + fallback)
├── ffmpeg_util.py               # Shared ffmpeg utilities
├── nodes/
│   ├── dream_factory.py         # FusionMLXDreamFactory ComfyUI adapter
│   ├── prompt_expand.py         # Legacy HTTP node
│   ├── flux_image.py            # Legacy HTTP node
│   ├── horror_tts.py            # Legacy HTTP node
│   ├── ken_burns.py             # Legacy HTTP node
│   └── assemble.py              # Legacy HTTP node
├── pipeline/
│   ├── engine.py                # PipelineEngine
│   ├── context.py               # PipelineContext
│   ├── stage.py                 # Stage ABC + StageInfo
│   ├── model_manager.py         # ModelManager + ModelHandle
│   ├── checkpoint.py            # CheckpointManager
│   ├── result.py                # PipelineResult
│   ├── stages/
│   │   ├── story_ingest.py      # PDF/EPUB/TXT ingestion
│   │   ├── prompt_expand.py     # LLM prompt expansion
│   │   ├── image_generate.py    # Flux image generation
│   │   ├── tts_synthesize.py    # TTS synthesis
│   │   ├── ken_burns.py         # Ken Burns + VideoToolbox
│   │   ├── assemble.py          # Clip concatenation + VideoToolbox
│   │   └── digital_human_render.py  # Digital human (fallback)
│   └── templates/
│       ├── short_drama.yaml
│       ├── ad_drama.yaml
│       ├── puppet_show.yaml
│       ├── medium_video.yaml
│       ├── series.yaml
│       ├── digital_human.yaml
│       └── digital_human_live.yaml
├── prompts/
│   ├── horror_director.md
│   ├── ad_director.md
│   ├── puppet_director.md
│   ├── medium_director.md
│   ├── series_director.md
│   └── digital_human_director.md
└── tests/
    ├── test_pipeline_engine.py
    ├── test_model_manager.py
    ├── test_checkpoint.py
    ├── test_stages.py
    ├── test_new_stages.py
    └── ... (legacy node tests)
```

## Next.js UI

The Dream Factory panel in the openclaw-mission-macos dashboard:

- Content type selector (7 types)
- Story seed input + story file upload (PDF/EPUB/TXT for series mode)
- Digital human avatar reference input
- Parameter panel (scene count, style preset, etc.)
- Real-time progress with stage + scene tracking
- Video player for completed output
- Job history with resume support

API routes: `/api/comfyui/dream-factory/{run,status,resume,jobs,upload}`

## Performance (M5 Max 128GB)

| Phase | Model | GPU Peak | Time (8 scenes) |
|-------|-------|----------|-----------------|
| A: LLM | Qwen3.5-9B | 5.6G | ~10s |
| B: Image | Flux | 7.0G | ~4-8min |
| C: TTS | Qwen3-TTS | 2.9G | ~2min |
| D: FFmpeg | None | 0G | ~30s (VideoToolbox) |
| **Total** | | **7.0G** | **~7-11min** |

30-min series (40 scenes): ~50-85min GPU time, checkpoint/resume essential.

## License

MIT
