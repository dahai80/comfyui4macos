# P3.3 AI 视频运动评估

> 评估日期：2026-07-06 ｜ 评估人：自动落地 ｜ 状态：评估完成，结论=**暂不建真视频引擎，延续"动作幻觉"路线，持续观测 MLX 视频生态**

## 一、目标

P0-P2 已用 **多姿态关键帧 + Ken Burns + 硬切剪辑** 实现"动作幻觉"（定格/分镜风格），P3.3 评估是否引入**真 image-to-video**（单图→短视频，模型生成帧间运动）以补"真动作"。

## 二、现状盘点（实测，非推测）

| 维度 | 现状 | 证据 |
|:---|:---|:---|
| fusion-mlx 视频端点 | **无** `/v1/video`，仅有 `/v1/images` | `fusion_mlx/api/images.py`，全仓 grep 无 video 生成路由 |
| fusion-mlx 视频引擎 | **无** VideoGenEngine | engine_pool 仅注册 LLM/Audio/ImageGen |
| mflux 能力 | **仅图像**（Flux1/Redux/InContext/Kontext/ControlNet/Depth/Fill/ConceptAttention，8 变体均 image-only） | mflux 源码 |
| 本地 HF 缓存视频模型 | **0 个**（仅 Flux-1.lite-8B / Qwen3.x / Qwen3-TTS） | `~/.cache/huggingface/hub` 实测 |
| MLX 原生视频扩散生态 | **极薄/实验性**（本环境 web search 不可用，无法在线核实 2026-07 最新；按已知基线：Wan2.1/CogVideoX/HunyuanVideo/LTX-Video/Mochi 均为 PyTorch 主线，MLX 移植稀少且多为实验） | 待另一组在线核实 |

**结论**：当前 MLX 栈无可用视频生成能力，真 image-to-video 在本机**不可直接落地**。

## 三、三条可选路径

### 路径 A：建 fusion-mlx 视频引擎（MLX 原生）
- 依赖一个稳定的 MLX 移植视频模型（待核实是否存在）。
- 工作量：新引擎 + `/v1/video/generate` 路由 + 池管理 + ComfyUI 侧 `VideoGenStage`。
- 优点：守住"MLX 原生、无 torch、同 venv"原则。
- 风险：模型生态不成熟→可能无米下锅；视频扩散显存/时间成本远高于图像（单段 2-4s 视频在 M 系列芯片可能 30s-数分钟），短剧批量生产成本陡增。

### 路径 B：非 MLX 路径（PyTorch 视频模型 / 远程 API）
- 本地另起 torch venv 跑 Wan/CogVideoX，或调远程视频 API。
- 优点：模型成熟、效果可控。
- 风险：**违背项目核心约束**（"仅 HTTP 桥到 fusion-mlx，无 in-process torch"，torch-vs-mlx venv 冲突正是 fusion-mlx 存在的理由）；引入第二条服务链路，运维复杂度翻倍。**不推荐**。

### 路径 C：延续"动作幻觉"路线（现状）
- 沿用 P3.1 `MultiPoseStage`（同角色 N 姿态 + Ken Burns + 硬切 + 旁白）。
- 优点：已落地、593→608 passed、零新依赖、成本可预测。
- 缺点：无真连续运动，定格/分镜质感（对民间鬼故事短剧**反而是合适美学**——剪纸/定格恐怖感）。

## 四、推荐

**采纳路径 C，路径 A 列为长期观测项，否决路径 B。**

理由（Rule 2 简单优先 + Rule 4 目标驱动）：
1. **美学契合**：鬼故事短剧的"定格/分镜/纸人"质感与多姿态+Ken Burns 天然匹配，真连续运动反可能削弱恐怖氛围。
2. **成本/收益**：真视频引擎开发+推理成本高，而当前路线已满足"动作"叙事需求（P2 验收通过）。
3. **生态未成熟**：MLX 视频模型稀少，贸然建引擎可能无可用模型。
4. **守住约束**：否决路径 B 避免引入 torch 第二链路。

## 五、对 fusion-mlx 的要求（PR 候选，**暂不提**）

若未来路径 A 启动（前提：经另一组在线核实存在稳定 MLX 视频模型），再向 fusion-mlx 提 PR：

- **需求**：新增 `VideoGenEngine` + `POST /v1/video/generate`（image-to-video），入参 `image`(b64/URL)+`prompt`+`duration`+`fps`+`seed`+`motion_strength`，出参 `b64_json`/`url`（mp4）。
- **目标结果**：固定 image+seed 可复现；2-4s/24fps mp4；显存自适应降级；FUSION_MLX_API_KEY 鉴权 + SSRF 防护（同 PR #31 安全规范）。
- **触发条件**：MLX 视频模型可用性核实通过 + 用户明确要"真动作"。

**当前不提此 PR**（前提未满足，避免给另一组空活）。

## 六、落地动作

- [x] 评估完成，结论归档（本文件）
- [x] ROADMAP P3.3 标注"评估完成→延续动作幻觉"
- [ ] 长期：每季度观测 MLX 视频生态，满足触发条件再启路径 A
