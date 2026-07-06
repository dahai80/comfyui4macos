# P3.5 多智能体编排评估

> 评估日期：2026-07-06 ｜ 状态：评估完成，结论=**不引入多智能体编排，保留 PipelineEngine + 分层故事结构**

## 一、评估对象

ROADMAP 原列 P3 候选："多智能体编排评估"——是否引入 Lore/Script/Prompt/Audio-Video 专员智能体（multi-agent）替代或增强当前 `PipelineEngine` 线性流水线。

## 二、现状（实测）

- `pipeline/engine.py`：`_STAGE_REGISTRY` 注册式阶段，`run()` 按序实例化执行，支持 `resume`（按 `completed_stages` 断点续跑）、中间产物清理、`_FINAL_STAGE` 按 content_type 收束。
- 15 个 stage（story_ingest→prompt_expand→image_generate→ken_burns/multi_pose→tts_synthesize/voice_clone→sfx→subtitle→assemble→publish→…），分层故事结构（summary→outline→prompt_expand 逐集场景）已在 P1 落地。
- 配置驱动 4×2 矩阵（character_style × motion_mode × duration_mode），engine 按 config 选 stage 组合。
- P0 设计明确："多智能体 P0 不做，用 PipelineEngine + 分层故事结构替代，P3 再评估"。

## 三、多智能体会带来什么

**潜在收益**：
- 专员智能体并行（Lore 建世界观 / Script 写本 / Prompt 工程视觉 / Audio-Video 组装）。
- 智能体间迭代精修（emergent 创意）。

**实际成本**：
| 成本 | 说明 |
|:---|:---|
| 非确定性 | 智能体对话=控制流交给模型，违反 Rule 5（能用代码确定的别交给模型）；产出难复现，破坏短剧批量生产的可验证性 |
| Token 爆炸 | 智能体间消息传递使 LLM 调用量乘性增长，单集成本不可控 |
| 测试困难 | Rule 9：多智能体交互路径组合爆炸，测试要么形同虚设要么维护成本极高 |
| 复杂度 | Rule 2：当前线性流水线已捕获"分解+专精"收益，多智能体是为一两个边际场景引入整套编排框架，过度工程 |
| 可观测性 | 线性 stage 日志清晰可定位；多智能体对话追踪需额外 trace 设施 |

## 四、当前架构已覆盖多智能体的"收益"

| 多智能体卖点 | 当前实现 | 是否足够 |
|:---|:---|:---|
| 任务分解 | 15 stage 线性 + 分层故事（summary→outline→scene） | ✅ |
| 角色专精 | 每 stage 单一职责（prompt_expand 专做视觉 prompt，tts 专做语音） | ✅ |
| 迭代精修 | story_ingest 摘要→大纲→逐集 prompt_expand 已是多轮 LLM 精修 | ✅ |
| 并行 | stage 间天然顺序依赖；并行仅在 stage 内（如 multi_pose N 姿态）已实现 | ✅ |
| 可复现 | 固定 seed + 线性 stage + checkpoint → 完全可复现 | ✅（多智能体反而丢失） |

**结论**：多智能体的结构性收益已被"线性 stage + 分层故事"等价捕获，且保留了确定性、可测试、可复现——短剧生产线最看重的三件事。

## 五、推荐

**否决多智能体编排，保留 PipelineEngine。** 仅在以下任一触发条件满足时再评估：

1. 需要**跨集角色记忆/世界观演化**的开放式叙事（当前 series_orchestrate + character_registry 跨集一致已够用）。
2. 需要**模型自主决策剧情走向**（如观众反馈驱动的分支剧本）——但那也只需在 story_ingest 加一个"反馈→大纲"的 LLM 步骤，无需整套多智能体框架。
3. 单 stage 内出现**真正需要多轮对话协商**的子任务（目前无）。

## 六、落地动作

- [x] 评估完成，结论归档（本文件）
- [x] ROADMAP P3.5 标注"评估完成→不引入多智能体"
- [ ] 触发条件出现时再启评估
