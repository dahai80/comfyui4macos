# custom_nodes4macos

ComfyUI custom nodes that bridge to **fusion-mlx** (native MLX inference daemon on Apple Silicon). Part of the `comfyui4macos` project — turning folk ghost stories into vertical short-drama episodes.

设计原则：ComfyUI 当流程壳，fusion-mlx 当模型底座，本包只做 HTTP 桥接 + 鬼故事领域逻辑。不 fork、不重写推理代码。

## 当前节点

| 节点 | 类名 | 说明 | 阶段 |
|---|---|---|---|
| FusionMLX Prompt Expand (Horror Director) | `FusionMLXPromptExpand` | 故事种子 → 结构化分镜 JSON（visual_prompt / audio_script / sound_effect / duration） | Phase 1 ✅ |
| FusionMLX Flux Image (Horror Visual) | `FusionMLXFluxImage` | visual_prompt + global_style → IMAGE 张量（torch 延迟导入） | Phase 2 ✅ |
| FusionMLX Horror TTS (Eerie Narration) | `FusionMLXHorrorTTS` | audio_script + instructions → wav 文件路径 | Phase 2 ✅ |
| FusionMLX Ken Burns (Still→9:16) | `FusionMLXKenBurns` | 单图 IMAGE + 可选旁白 → 9:16 mp4（zoompan 推镜，时长跟音轨或手填） | Phase 3 ✅ |
| FusionMLX Assemble (Clips→Drama) | `FusionMLXAssemble` | 多段 mp4 路径 → 9:16 单片（concat + 可选 BGM + 可选淡入淡出） | Phase 3 ✅ |

后续阶段：openclaw 编排（分镜→批量出图/TTS→KenBurns→Assemble 全自动）+ 自动发布。

## 前置条件

1. **fusion-mlx 已安装并运行**（`~/claude-home/fusion-mlx`，CLI `fusion-mlx` 或 `fm`）。
   - 默认端口 **11434**，host 127.0.0.1（见 `~/.fusion-mlx/settings.json`）。
   - `/health` 不需要鉴权；`/v1/*` 需要 API key（settings.json `auth.api_key`）。
2. **本机端口避让**：8000 被 `finance-api`（launchd 常驻）占用；fusion-mlx 走 11434，不冲突。
3. **ComfyUI** 已就绪（`~/solution/comfyui4macos/ComfyUI`，0.27.0）。ComfyUI 的 venv 需含 `httpx`（默认有）；图像节点在运行时才 `import torch` / `PIL`（ComfyUI 自带）。
4. **FFmpeg**（KenBurns / Assemble 用）：`brew install ffmpeg`。需 `ffmpeg` + `ffprobe` 在 PATH（或用 `FFMPEG_BIN` / `FFPROBE_BIN` 覆盖）。KenBurns 的 IMAGE→PNG 走 `numpy` + `PIL`（系统/ComfyUI 均有）。

## 安装（让 ComfyUI 发现本包）

在 ComfyUI 的 custom_nodes 目录建软链（保持本仓库在 comfyui4macos 根，不污染 ComfyUI 源码）：

```bash
ln -s ~/solution/comfyui4macos/custom_nodes4macos \
      ~/solution/comfyui4macos/ComfyUI/custom_nodes/custom_nodes4macos
```

启动 ComfyUI，日志应出现：
```
[custom_nodes4macos] INFO loading custom_nodes4macos
[custom_nodes4macos] INFO registered node FusionMLXPromptExpand
[custom_nodes4macos] INFO registered node FusionMLXFluxImage
[custom_nodes4macos] INFO registered node FusionMLXHorrorTTS
[custom_nodes4macos] INFO registered node FusionMLXKenBurns
[custom_nodes4macos] INFO registered node FusionMLXAssemble
```

## 配置

节点优先读环境变量，也可在节点 `base_url` / `api_key` 输入框覆盖：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `FUSION_MLX_BASE_URL` | `http://localhost:11434` | fusion-mlx 服务地址 |
| `FUSION_MLX_API_KEY` | （空） | fusion-mlx 开了 auth 必填（见 settings.json `auth.api_key`） |
| `FUSION_MLX_TIMEOUT` | `120` | 请求超时秒 |
| `FUSION_MLX_RETRIES` | `2` | 5xx / 网络错误重试次数 |
| `CUSTOM_NODES4MACOS_LOG_LEVEL` | `INFO` | 本包日志级别 |
| `FFMPEG_BIN` / `FFPROBE_BIN` | PATH 自动探测 | KenBurns / Assemble 用的 ffmpeg/ffprobe 路径 |
| `FFMPEG_TIMEOUT` | `300` | ffmpeg 单次调用超时秒 |

> 运行 ComfyUI 前导出 key：`export FUSION_MLX_API_KEY=<your-key>`，否则所有 `/v1/*` 调用 401。

## 模型选择

- 节点 `model` 下拉首项为 `(auto)`：
  - **Prompt Expand**：`(auto)` → fusion-mlx `/health` 的 `default_model`。复杂 schema 建议手选 9B+ 模型（0.6B 难以稳定输出 JSON schema）。
  - **Horror TTS**：`(auto)` → 从 `/v1/models` 里挑名字含 `tts` 的模型（如 `Qwen3-TTS-12Hz-1.7B-Base-8bit`）；找不到则回退 `tts-1`。
  - **Flux Image**：`(auto)` → 服务端 images 路由默认 `flux-2`。**需要 fusion-mlx 实例挂载 images 路由并加载 flux 模型**（见下「运行实例说明」）。
- 下拉列表来自 `/v1/models`，缓存 60s；服务未起或 401 时退化为只有 `(auto)`。

## 运行实例说明（重要）

本机 11434 上跑的是 **oMLX.app**（`/Applications/oMLX.app` 的 `omlx-server`，launchd 自启），**不是** `~/claude-home/fusion-mlx` 源码构建。差异：

| 能力 | oMLX.app（当前） | dev fusion-mlx 源码 |
|---|---|---|
| `/v1/chat/completions`（LLM） | ✅ | ✅ |
| `/v1/audio/speech`（TTS） | ✅（`Qwen3-TTS-12Hz-1.7B-Base-8bit`） | ✅ |
| `/v1/images/generate`（Flux 图像） | ❌ 路由未挂载（OpenAPI 无此路径） | ✅（`server.py` 无条件挂载 images_router） |

- 磁盘上有 `Flux-1.lite-8B-MLX-Q4` 模型，但 oMLX 不暴露图像路由 → **图像节点在当前实例上不可用**，live 测试自动 skip。
- 要启用图像生成：从 `~/claude-home/fusion-mlx` 源码起 dev 服务（含 images 路由 + flux 引擎依赖）到空闲端口，并把节点 `base_url` 指过去；或等 oMLX 后续版本支持。

## 使用

1. 启动 fusion-mlx（默认 11434）。
2. `export FUSION_MLX_API_KEY=<key>` 后启动 ComfyUI。
3. 画布串联：
   - `FusionMLX Prompt Expand` → 输出 `scenes_json` + `scene_count`
   - `FusionMLX Flux Image` ← 取 `visual_prompt` + `global_style` → `IMAGE`
   - `FusionMLX Horror TTS` ← 取 `audio_script` + `instructions` → `audio_path`
   - `FusionMLX Ken Burns` ← `IMAGE` + `audio_path`（可选）→ `video_path`（单镜 9:16 mp4）
   - `FusionMLX Assemble` ← 多个 `video_path`（每行一个）+ 可选 `bgm_path` → 成片 `video_path`

`scenes_json` 示例：
```json
{
  "story_title": "破庙借火",
  "global_style": "Chinese ink-wash dark fantasy, ...",
  "scenes": [
    {"scene_id": 1, "visual_prompt": "...", "audio_script": "...", "sound_effect": "wind howling", "duration_seconds": 5}
  ]
}
```

> 容错：若弱模型返回裸 JSON 数组（非对象），`Prompt Expand` 会自动包成 `{"scenes": [...]}` 并告警；建议换更强模型。

## 测试

```bash
cd ~/solution/comfyui4macos/custom_nodes4macos
python3 -m pytest -v                  # 离线测试（无需 fusion-mlx，CI 友好）
FUSION_MLX_API_KEY=<key> python3 -m pytest -v   # 含 live 端到端
python3 -m pytest -v -m live          # 仅 live（需 fusion-mlx 运行）
```

- 离线测试覆盖节点逻辑（mock 客户端），不依赖 torch / fusion-mlx。
- `live` 测试在 fusion-mlx 未运行、未鉴权、或对应能力（图像）未挂载时自动 skip。
- KenBurns / Assemble 测试用真实 ffmpeg 跑极小素材（64x128、<1s），无 ffmpeg 时自动 skip；不依赖 torch / fusion-mlx。
- 实测（oMLX.app @ 11434 + key）：chat live ✅、TTS live ✅（16s 生成 wav）、image live ⏭ skip（oMLX 无 images 路由）；KenBurns / Assemble 离线 ✅（ffmpeg 8.1.1）。全套 **34 passed + 1 skipped**。

## 目录

```
custom_nodes4macos/
├── __init__.py            # 节点注册（try/except 隔离失败）
├── fusion_client.py       # fusion-mlx HTTP 客户端 + list_models_safe/default_model_safe
├── ffmpeg_util.py         # ffmpeg/ffprobe subprocess 封装（run_ffmpeg/probe_duration/probe_has_audio）
├── nodes/
│   ├── prompt_expand.py   # 提示词扩展节点
│   ├── flux_image.py      # Flux 图像节点（torch 延迟导入）
│   ├── horror_tts.py      # 恐怖旁白 TTS 节点
│   ├── ken_burns.py       # 单图 Ken Burns → 9:16 mp4（zoompan，不依赖 fusion-mlx）
│   └── assemble.py        # 多段 mp4 concat → 成片（不依赖 fusion-mlx）
├── prompts/horror_director.md  # 系统 prompt
├── tests/                 # pytest（离线 + live）
├── workflows/             # 参考 .json 工作流（后续）
├── web/                   # 前端 JS（后续）
└── pytest.ini
```
