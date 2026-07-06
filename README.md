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
  │     ├── StoryIngestStage     — PDF/EPUB/TXT → chapter split → episode outline (LLM via HTTP)
  │     ├── PromptExpandStage    — story seed → structured scene JSON (LLM via HTTP)
  │     ├── ImageGenerateStage   — visual_prompt → PNG (Flux via fusion-mlx HTTP /v1/images/generate)
  │     ├── VoiceCloneStage      — ref_audio → ref_text (Whisper via HTTP /v1/audio/transcriptions)
  │     ├── TTSSynthesizeStage   — audio_script → WAV (Qwen3-TTS via HTTP /v1/audio/speech) + auto duration
  │     ├── KenBurnsStage        — PNG + audio → mp4 clip (ffmpeg + VideoToolbox)
  │     ├── AssembleStage        — clips → final mp4 (ffmpeg concat + VideoToolbox)
  │   └── DigitalHumanRenderStage — avatar + audio → video (fallback: static composite)
  ├── ModelManager    — RemoteHandle over FusionMLXClient (acquire → use → release; models live in fusion-mlx)
  ├── CheckpointManager — stage + scene-level checkpointing, resume from any point
  └── PipelineContext  — file-based artifact store (PNG/WAV/MP4 on disk)
```

### Publishing to Douyin draft box (opt-in)

`PublishStage` + `custom_nodes4macos/publisher/` (fork-and-strip of social-auto-upload, Douyin only) uploads the final video to the Douyin creator draft box. It is **off by default** and never crashes the pipeline.

Enable via config override (e.g. in `run` route or `engine.run` kwargs):

```python
stages=["story_ingest", "series_orchestrate", "publish"],  # append publish
publish_enabled=True,
publish_dry_run=True,            # True = write manifest only, no network (safe default)
publish_cookies_path="/abs/cookies.json",  # required when dry_run=False
publish_title="画皮", publish_tags=["鬼故事","民间传说"],
publish_all_episodes=True,       # series: iterate _episode_finals (one draft per episode)
```

`publish_dry_run=False` lazily imports Playwright (`pip install playwright && playwright install chromium`) and drives `creator.douyin.com/content/upload` — set input file, fill title/tags, click 存草稿. Missing cookies → `PublishConfigError`; missing Playwright → `PublishDependencyError`; both degrade to `skipped` in the manifest. Browser-launch failures degrade to `failed`. Result is written to `publish_manifest.json` per job.

### Multi-pose stop-motion (`motion_mode=multi_pose`)

`MultiPoseStage` is a motion stage (alternative to `ken_burns`) that produces the same `clip` artifact, so `assemble` is untouched. Per scene it generates N character pose keyframes via Flux HTTP using the **same scene seed + character appearance** (consistency) but **different pose-suffix prompts** (action), then stitches them into one clip: each pose gets a short Ken Burns zoom segment, hard-cut concatenated (stop-motion/storyboard aesthetic), muxed with the scene narration audio. This delivers the "动作幻觉" style for `卡通动作` / `真实人物形象` without any AI video engine (true video motion remains blocked by the no-MLX-video-engine constraint).

Enable via config override — the engine auto-swaps `ken_burns` → `multi_pose` when `motion_mode=multi_pose`:

```python
engine.run("puppet_show", motion_mode="multi_pose",
           multi_pose_count=3,                # keyframes per scene (>=2)
           multi_pose_poses=["facing forward", "turning side", "reaching out"])  # optional override
```

Pose generation failures degrade gracefully (reuse the base image for that pose); cached pose PNGs are reused on resume. Default pose suffixes and Ken Burns dimensions/fps come from the template (`ken_burns_width`/`ken_burns_height`/`ken_burns_fps`).

#### Realistic character identity (`character_style=realistic`)

For the `真实人物形象` track, set `character_style=realistic`. Two layers of reference-image conditioning then flow to fusion-mlx `/v1/images/generate` (fields `reference_image` / `reference_strength` / `conditioning_mode`):

- **Cross-scene identity** — `image_generate` resolves each scene's primary character's `reference_image` from `character_registry` (absolute path, or relative to `character_reference_dir`) and passes it as the redux reference for that scene's base image, so the same character looks the same across every scene.
- **Within-scene pose consistency** — `multi_pose` uses the scene's base image as the redux reference for all N pose keyframes, so poses vary without "换脸".

```python
engine.run("puppet_show", character_style="realistic", motion_mode="multi_pose",
           character_reference_dir="/abs/path/to/refs",
           character_registry=[
               {"name": "lao_wang", "appearance": "old man, grey hair",
                "reference_image": "lao_wang.png"},  # canonical face photo
           ],
           realistic_reference_strength=0.6,        # 0.0–1.0
           realistic_conditioning_mode="redux")      # "redux" (default) or "in_context"
```

> **Dependency:** identity preservation requires fusion-mlx PR #31 (`/v1/images/generate` reference-image conditioning via mflux `Flux1Redux` / `Flux1InContextDev`). Until it lands, the reference fields are sent but ignored by fusion-mlx (pydantic drops unknown fields), so output is plain txt2img — no errors, just no identity lock. Spec: `dahai80/fusion-mlx` PR #31.

### HTTP-only inference (no in-process MLX)

All LLM / image / TTS / transcription calls go through `FusionMLXClient` (HTTP to fusion-mlx on port 11434). Stages hold a `RemoteHandle(handle.client, handle.model_name)` — they never `import mlx_lm` / `mflux` / `mlx_audio`. fusion-mlx owns model loading, eviction, and scheduling; this codebase only does HTTP bridging + ghost-story domain logic. No fallback path: if fusion-mlx is unreachable, `_acquire_handle` logs a warning and the call fails loudly (Rule 12).

### Models (loaded by fusion-mlx, not in-process)

| Model | Size | Purpose |
|-------|------|---------|
| Qwen3.5-9B-4bit | 5.6G | Prompt expansion + story ingestion |
| Flux-1.lite-8B-MLX-Q4 | 7.0G | Image generation (loaded on-demand by fusion-mlx images router) |
| Qwen3-TTS-12Hz-1.7B-Base-8bit | 2.9G | Text-to-speech (ICL voice clone fallback) |
| Whisper-Large-V3-Turbo | ~1G | Auto-transcribe ref_audio → ref_text when user omits it |

**On-demand loading**: fusion-mlx loads/evicts models per request; this pipeline never pins them in-process, so peak memory is fusion-mlx's concern, not the engine's.

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
| HTTP-only inference | All LLM/image/TTS/transcription via `FusionMLXClient`; no in-process MLX to load/evict in the engine |
| Explicit `max_tokens` | Stages pass `max_tokens=16384` on chat calls (fusion-mlx default 2048 truncates long scene JSON) |
| Auto audio duration | TTS output probed for actual duration → accurate KenBurns clips |
| Scene-level checkpointing | Resume mid-stage without re-processing completed scenes |
| Scene-level idempotency | Each stage checks `has_artifact_on_disk()` before generating |
| Global scene_id for series | `_renumber_scenes()` ensures unique IDs across episodes |
| global_style propagation | LLM output `global_style` extracted and passed to ImageGenerateStage |
| Consolidated VideoToolbox | All stages use `ffmpeg_util.video_encoder_args()` — single source of truth |
| One-time health check | `ModelManager._acquire_handle` pings fusion-mlx `/health` once per manager; warns (does not crash) if unreachable |
| Model shutdown | `ModelManager.shutdown()` closes the HTTP client after pipeline completes (or on failure) |
| Character consistency | `character_registry` tracks appearance + voice across scenes/episodes; injected into visual_prompt and TTS instructions |
| Seed-per-character | Same character name → deterministic hash offset → visual consistency across scenes |
| Cross-episode registry | Episode 1 defines character registry → carried forward to subsequent episodes via user message |
| Voice-gender alignment | Female characters auto-get female TTS voice via `_get_scene_instructions` gender inference |
| Voice cloning | Qwen3-TTS ICL via ref_audio/ref_text over HTTP; Whisper auto-transcribe when ref_text omitted; emotion tags `[laughing]` `[excited]` `[whisper]` |
| Chinese face default | Chinese content types auto-enforce `Chinese face, East Asian features` in character appearances |
| Visual-audio alignment | Prompt templates enforce `visual_prompt` must precisely depict `audio_script` actions |
| Friendly output naming | Final video gets human-readable filename: `故事标题_第X集.mp4` |

> Note: MLX-side optimizations (FlowMatchEuler scheduler, `mx.compile`, encoder eviction, tight denoising loop, cache management) now live inside fusion-mlx, not this codebase.

## Quick Start

```bash
# Setup
cd comfyui4macos
python3 -m venv .venv && source .venv/bin/activate
pip install Pillow numpy scipy pyyaml httpx opencv-python

# Optional: PDF/EPUB support
pip install pymupdf ebooklib beautifulsoup4

# Required at runtime: fusion-mlx running on port 11434 (loads MLX models on-demand)
# export FUSION_MLX_API_KEY=<key>

# Run tests (all mock, no fusion-mlx required)
PYTHONPATH=. python -m pytest custom_nodes4macos/tests/ -v

# Live run (requires fusion-mlx running + FUSION_MLX_API_KEY)
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
│   ├── model_manager.py         # ModelManager + RemoteHandle + ModelMode (HTTP, no in-process MLX)
│   ├── checkpoint.py            # CheckpointManager + CheckpointData
│   ├── result.py                # PipelineResult dataclass
│   ├── stages/
│   │   ├── story_ingest.py      # PDF/EPUB/TXT ingestion + chapter splitting (LLM via HTTP)
│   │   ├── prompt_expand.py     # LLM prompt expansion (multi-episode support, via HTTP)
│   │   ├── image_generate.py    # Flux image generation via fusion-mlx HTTP
│   │   ├── tts_synthesize.py    # TTS synthesis via fusion-mlx HTTP (auto audio duration)
│   │   ├── ken_burns.py         # Ken Burns + VideoToolbox (parallel render)
│   │   ├── multi_pose.py        # Multi-pose stop-motion (motion_mode=multi_pose, alt to ken_burns)
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

> The benchmarks below were measured pre-refactor with in-process MLX. Post-refactor all inference goes through fusion-mlx HTTP, so per-call latency gains an HTTP round-trip but model load/evict is fusion-mlx's responsibility. Re-benchmark against your fusion-mlx instance for current numbers.

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
