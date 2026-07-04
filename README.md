# comfyui4macos

MLX-native AI content production pipeline for macOS. One-click short drama, series, digital human, and puppet show — all running locally on Apple Silicon.

**梦工厂 Dream Factory** — 7 content types, one pipeline engine, zero cloud dependency.

## Features

| Type | Chinese | Duration | Key Feature |
|------|---------|----------|-------------|
| Short Drama | 短剧 | 2-3min | Static images + Ken Burns + TTS narration |
| Ad Drama | 广告剧 | 1-5min | Brand placement + product showcase |
| Puppet Show | 木偶剧 | 5-15min | Character consistency + scene animation |
| Medium Video | 中短视频 | 30min | Checkpoint/resume + scene-level idempotency |
| Series | 连续剧 | 30 episodes × 30min | PDF/EPUB story ingestion → auto-split → 30 episodes |
| Digital Human | 数字人 | Variable | TTS + digital avatar + lip sync (fallback: static composite) |
| Digital Human Live | 数字人直播 | Real-time | Placeholder for future livestream mode |

## Architecture

```
PipelineEngine
  ├── Stage ABC (pluggable pipeline steps)
  │     ├── StoryIngestStage     — PDF/EPUB/TXT → chapter split → episode outline
  │     ├── PromptExpandStage    — story seed → structured scene JSON (LLM)
  │     ├── ImageGenerateStage   — visual_prompt → PNG (FluxPipeline MLX)
  │     ├── VoiceCloneStage      — ref_audio → voice profile (Fish S2 Pro zero-shot / Whisper auto-transcribe)
  │     ├── TTSSynthesizeStage   — audio_script → WAV (Fish S2 Pro / Qwen3-TTS ICL / mlx_audio) + auto duration
  │     ├── KenBurnsStage        — PNG + audio → mp4 clip (ffmpeg + VideoToolbox)
  │     ├── AssembleStage        — clips → final mp4 (ffmpeg concat + VideoToolbox)
  │     └── DigitalHumanRenderStage — avatar + audio → video (fallback: static composite)
  ├── ModelManager    — sequential model lifecycle (acquire → use → release + mx.clear_cache)
  ├── CheckpointManager — stage + scene-level checkpointing, resume from any point
  └── PipelineContext  — file-based artifact store (PNG/WAV/MP4 on disk)
```

### MLX Native Models

| Model | Size | Purpose |
|-------|------|---------|
| Qwen3.5-9B-4bit | 5.6G | Prompt expansion + story ingestion |
| Flux-1.lite-8B-MLX-Q4 | 7.0G | Image generation (8-step denoising, dev-variant) |
| Qwen3-TTS-12Hz-1.7B-Base-8bit | 2.9G | Text-to-speech (ICL voice clone fallback) |
| Fish-Audio-S2-Pro | ~6G | Zero-shot voice cloning + emotion tags (primary) |
| Whisper-Large-V3-Turbo | ~1G | Auto-transcribe ref_audio → ref_text when user omits it |

**Sequential loading**: Peak GPU memory = 7.0G (Flux only), not 15.5G (all three).

### Content Type Templates

Content types differ only by YAML config + prompt files. Zero Python code to add new types.

```
pipeline/templates/
├── short_drama.yaml          # 短剧
├── ad_drama.yaml             # 广告剧
├── puppet_show.yaml          # 木偶剧
├── medium_video.yaml         # 中短视频
├── series.yaml               # 连续剧 (PDF→30集)
├── digital_human.yaml        # 数字人
└── digital_human_live.yaml   # 数字人直播 (placeholder)
```

## Performance Optimizations

| Optimization | Details |
|-------------|---------|
| VideoToolbox h264 | Auto-detected hardware encoding on Apple Silicon |
| MLX warmup | `mx.zeros(1)` pre-initializes Metal before first model load |
| **FlowMatchEulerDiscrete scheduler** | Compiled step function + better sigma scheduling → same quality with 6 steps instead of 8 |
| **mx.compile() on transformer** | Kernel fusion on Flux transformer forward pass — 15-25% denoising speedup |
| **Prompt pre-encoding + encoder eviction** | Pre-encode all prompts, then evict T5+CLIP (~4-6GB freed) before denoising |
| **Custom tight denoising loop** | Bypasses `Flux1.generate_image()` overhead (tqdm, callbacks, Config re-creation) |
| **Removed per-scene mx.clear_cache()** | Was forcing GPU buffer reallocation (~2-3s overhead per image); now only on model release |
| Seed variation | `flux_vary_seed` offsets seed per scene for diverse images |
| Parallel Ken Burns | Configurable worker threads (`ken_burns_workers: 2-3`) |
| Auto audio duration | TTS output probed for actual duration → accurate KenBurns clips |
| Auto memory budget | Detects system RAM via `sysctl hw.memsize`, reserves 40% + 4G |
| Scene-level checkpointing | Resume mid-stage without re-processing completed scenes |
| Scene-level idempotency | Each stage checks `has_artifact_on_disk()` before generating |
| Global scene_id for series | `_renumber_scenes()` ensures unique IDs across episodes |
| global_style propagation | LLM output `global_style` extracted and passed to ImageGenerateStage |
| Consolidated VideoToolbox | All stages use `ffmpeg_util.video_encoder_args()` — single source of truth |
| Lazy MLX imports | All `import mlx_*` inside stage methods; no crash without MLX |
| HTTP fallback | `FusionMLXClient` when MLX not available |
| Robust FluxPipeline loading | Tries `flux.FluxPipeline` → `mlx_flux.FluxPipeline` with clear error message |
| Model shutdown | `ModelManager.shutdown()` releases all models + clears GPU cache after pipeline completes (or on failure) |
| Character consistency | `character_registry` tracks appearance + voice across scenes/episodes; injected into visual_prompt and TTS instructions |
| Seed-per-character | Same character name → deterministic hash offset → visual consistency across scenes |
| Cross-episode registry | Episode 1 defines character registry → carried forward to subsequent episodes via user message |
| Voice-gender alignment | Female characters auto-get female TTS voice via `_get_scene_instructions` gender inference |
| Voice cloning | Fish S2 Pro zero-shot voice cloning (5-10s ref_audio → cloned voice); Qwen3-TTS ICL fallback; Whisper auto-transcribe when ref_text omitted; emotion tags `[laughing]` `[excited]` `[whisper]` |
| Chinese face default | Chinese content types auto-enforce `Chinese face, East Asian features` in character appearances |
| Visual-audio alignment | Prompt templates enforce `visual_prompt` must precisely depict `audio_script` actions |
| Friendly output naming | Final video gets human-readable filename: `故事标题_第X集.mp4` |

## Quick Start

```bash
# Setup
cd comfyui4macos
python3 -m venv .venv && source .venv/bin/activate
pip install torch mlx mlx-lm mlx-audio Pillow numpy scipy pyyaml httpx

# Optional: PDF/EPUB support
pip install pymupdf ebooklib beautifulsoup4

# Run tests (all mock, no MLX required)
PYTHONPATH=. python -m pytest custom_nodes4macos/tests/ -v

# MLX live test (requires models in HuggingFace cache)
PYTHONPATH=. python -c "
from custom_nodes4macos.pipeline import PipelineEngine
engine = PipelineEngine()
result = engine.run('short_drama', story_seed='深夜赶路遇白衣女子')
print(result)
"

# Series from PDF
PYTHONPATH=. python -c "
from custom_nodes4macos.pipeline import PipelineEngine
engine = PipelineEngine()
result = engine.run('series', story_file='/path/to/novel.pdf', episode_count=30)
print(result)
"
```

## ComfyUI Integration

Single node `FusionMLXDreamFactory` encapsulates the entire pipeline:

- **Input**: `content_type` (7 types), `story_seed`, `story_file`, `episode_count`, `avatar_reference`, `config_overrides`
- **Output**: `video_path`, `scenes_json` (with progress tracking)
- **Resume**: Pass `resume_job_id` to continue from checkpoint

### Example Workflow JSON

```json
{
    "1": {
        "class_type": "FusionMLXDreamFactory",
        "inputs": {
            "content_type": "short_drama",
            "story_seed": "深夜赶路遇白衣女子，荒村古寺钟声起",
            "scene_count": 8,
            "style_preset": "水墨悬疑"
        }
    }
}
```

### Web Panel

Open the Dream Factory panel from ComfyUI → click "🎬 打开梦工厂" button on the node, or navigate to `/extensions/custom_nodes4macos/dream_factory.html`.

## File Structure

```
custom_nodes4macos/
├── __init__.py                  # Register FusionMLXDreamFactory + 5 legacy nodes
├── fusion_client.py             # HTTP bridge (legacy + fallback)
├── ffmpeg_util.py               # Shared ffmpeg utilities (VideoToolbox auto-detect)
├── nodes/
│   ├── dream_factory.py         # FusionMLXDreamFactory ComfyUI adapter
│   ├── prompt_expand.py         # Legacy HTTP node
│   ├── flux_image.py            # Legacy HTTP node
│   ├── horror_tts.py            # Legacy HTTP node
│   ├── ken_burns.py             # Legacy HTTP node (VideoToolbox)
│   └── assemble.py              # Legacy HTTP node (VideoToolbox)
├── pipeline/
│   ├── engine.py                # PipelineEngine (template merge + stage orchestration)
│   ├── context.py               # PipelineContext (file-based artifact store)
│   ├── stage.py                 # Stage ABC + StageInfo
│   ├── model_manager.py         # ModelManager + ModelHandle + ModelMode
│   ├── checkpoint.py            # CheckpointManager + CheckpointData
│   ├── result.py                # PipelineResult dataclass
│   ├── stages/
│   │   ├── story_ingest.py      # PDF/EPUB/TXT ingestion + chapter splitting
│   │   ├── prompt_expand.py     # LLM prompt expansion (multi-episode support)
│   │   ├── image_generate.py    # Flux image generation (per-scene cache clear)
│   │   ├── tts_synthesize.py    # TTS synthesis (auto audio duration)
│   │   ├── ken_burns.py         # Ken Burns + VideoToolbox (parallel render)
│   │   ├── assemble.py          # Clip concatenation + VideoToolbox
│   │   └── digital_human_render.py  # Digital human (static avatar fallback)
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
├── web/
│   ├── dream_factory.html       # Dream Factory web panel
│   └── dream_factory.js         # ComfyUI JS extension
├── workflows/
│   ├── dream_factory_short_drama.json
│   └── dream_factory_series_pdf.json
└── tests/
    ├── test_pipeline_engine.py
    ├── test_model_manager.py
    ├── test_checkpoint.py
    ├── test_stages.py
    ├── test_new_stages.py
    ├── test_performance_and_fixes.py
    └── ... (legacy node tests)
```

## Performance (M5 Max 128GB)

### Baseline (LinearScheduler, 8 steps, per-scene cache clear)

| Phase | Model | GPU Peak | Time (8 scenes) |
|-------|-------|----------|-----------------|
| A: LLM | Qwen3.5-9B | 5.6G | ~10s |
| B: Image | Flux | 7.0G | ~6-8min |
| C: TTS | Qwen3-TTS | 2.9G | ~2min |
| D: FFmpeg | None | 0G | ~30s (VideoToolbox) |
| **Total** | | **7.0G** | **~9-11min** |

### Optimized (FlowMatchEuler 6 steps, mx.compile, encoder eviction, tight loop)

| Phase | Model | GPU Peak | Time (8 scenes) |
|-------|-------|----------|-----------------|
| A: LLM | Qwen3.5-9B | 5.6G | ~10s |
| B: Image | Flux | 7.0G → ~4G* | ~4min (32s/image) |
| C: TTS | Qwen3-TTS | 2.9G | ~6min |
| D: FFmpeg | None | 0G | ~2min (VideoToolbox) |
| **Total** | | **7.0G** | **~12min** |

*Peak during denoising only; text encoders evicted (~4-6GB freed) before denoising starts.

### Series Benchmark: 青溪渡阴 6集 (48 scenes)

| Phase | Time | Notes |
|-------|------|-------|
| Story Ingest | ~3s | 2 chapters, 14451 chars |
| Prompt Expand | ~90s | 6 episodes, Qwen3.5-9B |
| Image Generate | ~5.5min | 48 images, 32s/image avg, mx.compile + FlowMatchEuler 6-step |
| TTS Synthesize | ~5.8min | 48 audio clips |
| Ken Burns | ~1.7min | 48 clips, VideoToolbox |
| Assemble | ~1.0min | Final 8.7min video |
| **Total** | **~14min** | 1080×1920 h264, 108.8MB |

30-min series (80 scenes/episode × 30 episodes): ~30-50min GPU time per episode, checkpoint/resume essential.

### Optimization Details

| Opt | Technique | Impact |
|-----|-----------|--------|
| 1 | Remove per-scene `mx.clear_cache()` | ~5-8% (eliminates ~2-3s buffer realloc per image) |
| 2 | FlowMatchEulerDiscrete + 6 steps | ~18-20% (25% fewer denoising steps, better sigma scheduling) |
| 3 | `mx.compile()` on transformer | ~12-18% (kernel fusion on forward pass) |
| 4 | Pre-encode prompts + evict T5/CLIP | ~5-8% (frees ~4-6GB for GPU cache) |
| 5 | Custom tight denoising loop | ~3-5% (eliminates tqdm/callbacks/Config overhead) |

## License

MIT
