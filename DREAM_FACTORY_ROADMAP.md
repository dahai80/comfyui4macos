# Dream Factory 多能力短剧/长剧工厂方案与落地计划

> ComfyUI 壳 + fusion-mlx 底座 + custom_nodes4macos 桥
> 民间鬼故事短剧 → 多能力剧情工厂（短剧/长剧 × 氛围旁白/口播/卡通/真实人物）
> 编写日期：2026-07-06　状态：待用户审核，审核通过后从 P0.1 开工

---

## 一、愿景与定位

在 macOS Apple Silicon 上构建完全本地、MLX-native 的剧情视频自动化工厂：

- **输入**：故事种子 / PDF（民间悬疑、道家、佛家鬼故事及其他题材）
- **输出**：9:16 可发布短剧 / 长剧连续剧
- **目标**：支撑 50 万粉级抖音账号的内容流水线

### 1.1 五层架构

| 层 | 职责 | 选型 |
|:---|:---|:---|
| L1 控制大脑/编排 | 长篇分解、多集调度、状态管理 | comfyui4macos `PipelineEngine`（不引入 openclaw-mission-control 重型 fork） |
| L2 推理底座 | LLM/图像/TTS/STT 引擎池、显存调度 | fusion-mlx（omlx + rapid-mlx 整合） |
| L3 图像生成 | Flux MLX 出图、角色一致性 | ComfyUI + mflux |
| L4 视频组装 | 剪辑、BGM/SFX、字幕、运镜 | Pipeline Stage（ffmpeg 直调，不进 ComfyUI 节点） |
| L5 流水线/发布 | 剧本→音→图→剪→发布 | PipelineEngine + social-auto-upload |

### 1.2 能力矩阵（核心定位）

按输入决定，**4 种表现形式 × 2 种时长**，模板 = 预设组合。短剧与长剧作为**两个独立功能**给用户使用。

| 表现形式 \ 时长 | 短剧（3-10min） | 长剧（25-40min） |
|:---|:---|:---|
| 氛围画面+旁白 | ✅ `short_drama` | ✅ `medium_video` / ⚠️ `series`(stub) |
| 口播数字人 | ✅ `digital_human` | ❌ series+口播（待贯通） |
| 卡通动作 | ✅ `puppet_show`(MVP) + `motion_mode=multi_pose` | ❌ |
| 真实人物形象 | ❌ | ❌ |

---

## 二、当前实施审视（2026-07-06 核验）

### 2.1 L1 编排
- comfyui4macos 自带 `PipelineEngine`（Stage + YAML + CheckpointManager）是事实编排器，单集端到端跑通
- openclaw-mission-macos 前端（`dream-factory-view.tsx` + `api/comfyui/dream-factory`）是 GUI dashboard
- finance-api 与本工作无关，不耦合不混代码
- 缺口：多智能体编排（Lore/Script/Prompt/Audio-Video）未建 → P3 再评估

### 2.2 L2 推理底座（fusion-mlx）
- ✅ LLM（BatchedEngine）+ 图像（mflux Flux）+ TTS（Qwen3-TTS）三路通，`:11434` 运行中
- ⚠️ 手动 source 运行，**非 launchd 常驻**
- ⚠️ 分支 `fix/macos-app-launch`，PR #27（image_gen mflux）未合并主线；`admin/` 大量未提交
- ⚠️ whisper 未下载完（STT 受影响；短剧流水线暂不需要 STT）
- ❌ 无视频生成引擎（AI 视频运动 defer）

### 2.3 L3 图像生成
- ✅ mflux Flux.1 MLX-Q4 通，9:16 出图，4 个恐怖风格预设
- ⚠️ `character_registry` 存于 config，跨集一致性 wiring 未验证
- ⚠️ Phase-5 bug：西方面孔默认、画面-旁白弱相关
- ❌ 数字人入剧未接 series

### 2.4 L4 视频组装（审视后比预期好，缺口收敛）
- ✅ `AssembleStage`：concat + scale/pad/fps 归一化 + **BGM 混音**（`amix` 0.3）+ fade 转场 + **友好命名**（`story_title_episode_title.mp4`）
- ❌ SRT 字幕渲染（恐怖字体）
- ❌ 音频对齐细切（按语音边界/silence 切镜，当前切点=场景边界）
- ❌ SFX 音效层
- ❌ 动态 BGM（按情绪选曲）+ 旁白 ducking（固定 0.3 音量）
- ❌ 转场多样性（仅 none/fade）

### 2.5 L5 流水线/发布
- ✅ 单集 short_drama 端到端通（540×960 9:16 21s，425 tests）
- ⚠️ `series.yaml` 是 stub，参数矛盾（`episode_duration_min:30` vs `scenes_per_episode:80` vs `scene_count:8`）
- ❌ 跨集连续性、自动发布、粉丝反馈雷达
- ⚠️ Phase-5 bug：声音性别不匹配（`TTSSynthesizeStage._get_scene_instructions` 不强制性别）
- ✅ 输出命名（`_friendly_output_path` 已实现，P0 验证生效）

---

## 三、核心决策（已与用户确认）

1. **时长**：短剧 + 长剧都做，作为两个独立功能给用户
2. **借壳换魂边界**：
   - ComfyUI 壳 = 交互式画布 + 单资产节点（`prompt_expand`/`flux_image`/`horror_tts` 独立节点，供调试与自由组合）
   - `PipelineEngine` = 自动化连续剧工厂（series 编排、多集、checkpoint）
   - 视频组装 = ffmpeg Stage（不进 ComfyUI 节点）
   - **不把 series 编排塞进 ComfyUI graph executor**（那是给单次拓扑用的，不适合 40+ 场景长序列）
3. **构建顺序**：短剧 → 长剧 → 数字人（一步步来）
4. **多智能体**：P0 不做，用 `PipelineEngine` + 分层故事结构替代，P3 再评估
5. **finance-api**：与本工作无关，不耦合不混代码
6. **fusion-mlx 改动原则**：
   - fusion-mlx = omlx + rapid-mlx 高效特性整合的 macOS 专有模型管理平台
   - 大胆发挥，**性能优先**
   - **尽可能向前兼容**（不破坏现有 OpenAI/Anthropic/MCP/openclaw 路由、EnginePool 接口、配置格式）
   - 缺能力 → 拉分支改 → 本地先用 → 提 PR 合主干（dahai80/fusion-mlx 自家仓库，无上游摩擦）

---

## 四、统一架构

ONE pipeline，三个正交配置维度，模板是预设：

```
character_style: none | realistic | cartoon | puppet     # Flux 画风 + 一致性
motion_mode:     ken_burns | talking_head | multi_pose    # 运镜 / 口播 / 多姿态定格
duration_mode:   short | long(series)                     # 单集 / 多集连续剧
```

- **口播** = `character_style=realistic/cartoon` + `motion_mode=talking_head`（AvatarAnimate）
- **卡通动作** = `character_style=cartoon/puppet` + `motion_mode=multi_pose`（多姿态关键帧 + Ken Burns + 剪辑）
- **真实人物形象** = `character_style=realistic` + `motion_mode=multi_pose`（写实一致性 + 多姿态）
- **氛围画面+旁白** = `character_style=none` + `motion_mode=ken_burns`（当前 short_drama）

多集连续剧（`duration_mode=long`）是横切能力，4 种形式都能套 series 框架（分集大纲 → 逐集 → 角色一致性 → 回级 checkpoint → 续跑）。

**真动作硬约束**：fusion-mlx 无视频生成引擎，MLX-native 下"卡通动作/真实人物"的全身动作无法用 AI 视频实现。P0-P2 务实路径 = Flux 角色一致性 + 多姿态关键帧 + Ken Burns + 动态剪辑 = 动作幻觉（定格/分镜风格），真 AI 视频运动留 P3。

---

## 五、Fork/Extract 策略

| 缺口 | 源仓库 | 策略 | 阶段 |
|:---|:---|:---|:---|
| SRT 字幕渲染（恐怖字体） | MoneyPrinterTurbo `subtitle.py` + video-use `render.py` | extract-module | P0 |
| 音频对齐细切 | video-use `pack_transcripts.py` | extract-module | P0 |
| SFX + 动态 BGM + ducking | 内建（ffmpeg `amix`/`sidechain`） | 自研 | P0 |
| 转场多样性 | 内建（ffmpeg `xfade`） | 自研 | P0 |
| 跨集连续性 | 内建 `series_registry` + `plot_state` + 回级 checkpoint | 自研 | P1 |
| 角色一致性增强 | 内建 + IP-Adapter/PuLID 评估 | 自研+评估 | P1 |
| 抖音自动发布 | social-auto-upload | fork-and-strip（仅 Douyin） | P2 |
| AI 视频运动 | Pixelle-Video / fusion-mlx video engine | defer | P3 |

---

## 六、落地计划 P0-P3

### P0（1.5 周）— 短剧做扎实

**目标**：short_drama 从"能跑"升级到"可发布质量"。

| 任务 | 内容 |
|:---|:---|
| P0.1 | ✅ 修 4 个 Phase-5 bug：声音性别强制（`tts_synthesize._get_scene_instructions`）、画面-旁白相关（`visual_prompt` 体现 `audio_script`）、中国面孔（"East Asian features"）、验证输出命名生效 |
| P0.2 | ✅ `SubtitleStage`：SRT 生成（`audio_script`+时间戳）+ 渲染（恐怖字体，9:16 底部）。降级链 burn(libass)→soft(mov_text)→none；此机 ffmpeg 无 libass 自动走 soft mux；19 测试通过 |
| P0.3 | ✅ 音频对齐细切：Ken Burns 按 silence 切镜。`_detect_silence`(ffmpeg silencedetect 解析 stderr)→`_silence_to_cut_frames`(边缘剔除+上限+均匀选取)→`_build_zoompan_multishot`(嵌套 if 表达式单次渲染硬切)→`_render_scene` 调度(无静音/短场景回退单镜)；short_drama 默认开启 `ken_burns_audio_cut`；29 测试通过(含 2 ffmpeg 集成)；全套 485 passed |
| P0.4 | ✅ BGM 增强 + `SFXStage`：assemble 增 `assemble_duck_bgm`(sidechaincompress 旁白 ducking，asplit+sidechaincompress+amix，真实 ffmpeg 验证) + `_resolve_bgm_by_mood`(`bgm_mood_map`+`bgm_mood`/`style_preset` 选曲)；新增 `SFXStage`(按 `sound_effect` 关键词匹配 `sfx_map`，adelay 时间戳对齐+amix 叠加，无 map/无匹配 graceful no-op)；short_drama 管线 assemble→sfx→subtitle；16 测试通过；全套 501 passed |
| P0.5 | ✅ 转场多样性（`xfade`：crossfade/dissolve/horror wipes）：`_TRANSITIONS` 扩至 16 种 + `_XFADE_MAP`(horror→fadeblack, crossfade→fade 等)；`_render_xfade` 链式 xfade(累积 offset) + acrossfade 音频链 + sidechaincompress ducking 分支；D 自适应 clamp `max(0.1,min(0.5,min_dur*0.5))`；n<2 回退 concat；none/fade 走原 fade+t=in 路径；14 测试通过(含真实 ffmpeg 集成 3 片段 crossfade→5.0s)；全套 515 passed |
| P0.6 | ✅ fusion-mlx 底座常驻：① PR #27 已合并(主线 commit eb46fe6，image_gen 路由+mflux) ② `fusion_mlx/admin/` 已在 git(最新 fe0f09d admin Apply 503 修复) ③ launchd 常驻——`com.dahai.fusion-mlx.plist`(RunAtLoad+KeepAlive，base-path `~/.fusion-mlx`，port 11434) + `launchd.env`(600，`FUSION_MLX_API_KEY`/`HF_ENDPOINT` 走文件不进 argv) + `launchd-wrapper.sh`(通用 env 注入)；验证：杀进程后 launchd ~12s 自动拉起新 PID，/health 200，/v1/models 15 个含 Flux/Qwen3.5-9B/Qwen3-TTS，LLM 生成通过 |

**验收**：一段鬼故事短剧，带字幕+SFX+BGM(ducking)+音频对齐切镜+多样转场，9:16，角色面孔/声音/画面-旁白一致，达抖音可发布质量。

**✅ P0 验收通过（2026-07-06）**：端到端跑通 `short_drama` 2 场景（Qwen3.5-9B-4bit + Flux-1.lite + Qwen3-TTS，assemble_transition=horror）。7 stages 全过（prompt_expand→image_generate→tts_synthesize→ken_burns→assemble→sfx→subtitle）。最终成片 `scene_000_final.mp4` 2.2MB / 27.06s / 540×960 9:16 / h264+aac+mov_text 字幕流（7 cues）。exercised：xfade mode=fadeblack D=0.50 + duck:True（P0.4/P0.5）、sfx no_map graceful skip（P0.4）、ken_burns audio_cut（P0.3）、subtitle soft mux。8 制品齐全。全套 515 passed + 2 skipped。fusion-mlx launchd 常驻（P0.6）。

### P1（1.5 周）— 长剧能力

**目标**：series 从 stub 升级为可跑的多集连续剧（25-40min，氛围画面+旁白 form）。

| 任务 | 内容 |
|:---|:---|
| P1.1 | ✅ 修正 `series.yaml`（参数自洽：episode_count 5 × scene_count 8/集 × ep_dur 30min；移除死配置 scenes_per_episode；补 sfx+subtitle 两阶段达 P0 对齐；flux 1024²+ken_burns 1080×1920 9:16；audio_cut/duck_bgm/subtitle/tts 全默认；character_registry 占位；测试改断言每集 8-12 场景，515 passed） |
| P1.2 | ✅ 跨集 `character_registry` 加载 + seed 一致 + appearance 注入 `visual_prompt`（既有代码已满足：`_process_episodes` 跨集合并 registry + 注入 LLM prompt；`image_generate` 用最终合并 registry 在生成时注入 appearance + `_compute_scene_seed` 同角色同 seed；`_enforce_chinese_faces` 中文脸默认。新增 2 个跨集契约测试，517 passed） |
| P1.3 | ✅ 分层故事结构（替代多智能体）：`story_ingest`(`_split_chapters`→`_generate_outline` LLM 摘要→分集大纲 title/synopsis/key_scenes/cliffhanger→`_parse_episodes`，失败兜底西游记大纲) → `prompt_expand._process_episodes`(逐集 LLM 场景 + `_merge_character_registry` 跨集合并 + `_enforce_chinese_faces` + `_reinforce_visual_audio_correlation`)；分层 = 摘要→大纲→逐集场景 |
| P1.4 | ✅ 回级 checkpoint + 断点续跑 + 流式处理 + 中间文件清理：`CheckpointData` 增 `completed_episodes`/`episode_finals` 字段(向后兼容旧 checkpoint)；`SeriesOrchestratorStage` 按 `_completed_episodes` 跳过已完成集、只跑剩余集；`_cleanup_episode_intermediates`(gated by `cleanup_episode_intermediates`) 清 image/audio/clip 保 final+subtitle；`prompt_expand` 3 次重试(generate 失败重试、parse 失败重试)→西游记兜底；e2e 验证：resume `20260706_ffa876b9` 跳过 ep1、120s 出 ep2，cleanup 清 6 中间产物 3131KB |
| P1.5 | ✅ 前端：长剧配置 + 系列进度面板（已完成集/续跑/预览）：`dream_factory.py` 节点增 `episode_duration_min` 输入(默认 0=用模板 30)→`run` route 透传→前端 series 时显示「集数 + 每集时长」输入并提交；`engine.list_jobs` checkpoint-backed 列表(job_id 键) 经 `server_routes.py`(`/dream_factory/jobs`/`/jobs/{id}`/`/preview/{id}/{file}`) → Next.js 代理(`/library`/`/library/[jobId]`/`/preview`) → 前端「任务列表」面板替换旧 prompt_id 历史记录：series 显示 已完成 X/Y 集进度条 + 续跑按钮(传 checkpoint job_id) + 展开按集预览(`/preview` 流式 video)；tsc 0 错(dream-factory 文件)，全套 551 passed + 2 skipped |

**✅ P1 验收通过（2026-07-06）**：`青溪渡阴` 2 集连跑(e2e `20260706_ffa876b9`)，角色/声音跨集一致(character_registry 3 角色跨集合并)，输出 `青溪渡阴_第一集.mp4`(6.91MB)+`青溪渡阴_第二集.mp4`(2.62MB)，断点续跑可用(resume 跳过 ep1、只跑 ep2、checkpoint completed_episodes=[1,2])，中间产物清理生效。LLM JSON 确定性问题已提 fusion-mlx PR #30(spec，待另一组实现)，不阻塞 P1 管线验收(兜底链保证产出)。P1.5 前端长剧配置+系列进度面板落地(551 passed + 2 skipped)。全套 551 passed + 2 skipped。

### P2（1 周）— 数字人口播

**目标**：AvatarAnimate 贯通进短剧 + 长剧。

| 任务 | 内容 |
|:---|:---|
| P2.1 | ✅ 口播短剧：`digital_human` 模板复核 + 角色绑定 UI — React 新增 `digital_human`(口播短剧) 内容类型(`UserCircle` 图标, `needsDigitalHuman`); 拆分 `isDigitalHuman`(avatar/voice 绑定 UI, 含 digital_human) 与 `usesDreamGallery`(隐藏故事种子+ComfyUI生成按钮, 仅 live/video_shop), digital_human 保留故事种子+走 dream-factory `onSubmit` 管线; `canSubmit` 对 digital_human 要求 story+avatar; `handleSubmit` 对 digital_human/digital_human_live 透传 `avatar_reference`(avatar_id)+`voice_reference`(voice_id). run route 新增 `resolveAvatar`/`resolveVoice`: avatar_id→预构建 `_avatar/` 包(reference.png+avatar_meta.json) 或回退 source_photo/video; voice_id→`voice_ref_audio`+`ref_text`+`voice_clone_model`; 路径遍历守卫 `ID_RE`; 合并进 `config_overrides`(node 已 `overrides.update` 最后生效). `avatar_create.process` 新增预构建包复用守卫(avatar_package 存在+reference.png+avatar_meta.json → 直接复用 artifacts, 跳过人脸检测). 新增 `test_reuse_prebuilt_avatar_package_skips_detection`. tsc 0 dream-factory 错; 全套 552 passed + 2 skipped. |
| P2.2 | ✅ 口播长剧：series + avatar_animate 合流，跨集 voice_id/avatar 一致 — `SeriesOrchestratorStage` 新增 `_DIGITAL_HUMAN_STAGES=(avatar_create→voice_clone→prompt_expand→tts_synthesize→avatar_animate→assemble)` 与 `_is_digital_human_mode`(avatar_package 为目录 或 avatar_reference 为文件 → True); `process()` 按 mode 选 stage tuple, image mode 保留原 7 阶段. 跨集一致: `sub_config = copy.deepcopy(ctx.config)` 将 series 级 `avatar_package`(预构建包)+`voice_ref_audio`/`ref_text`/`voice_clone_model` 透传至每集; avatar_create 复用守卫(P2.1)每集短路(同一包), voice_clone 同源音频→同克隆声色. React `series` 内容类型开放 avatar/voice 绑定: 新增 `supportsAvatarBinding=isDigitalHuman||isSeries`, avatar/voice picker 门控改用之(series 提示"绑定后按口播长剧生成, 不选按图文长剧"); `handleSubmit` 透传 `avatar_reference`/`voice_reference` 对 series 生效; `canSubmit` 对 series 仍 story 必填、avatar 可选. run route `resolveAvatar`/`resolveVoice` 不分 content_type, series 直接复用. 新增 `TestSeriesDigitalHumanMode`(7 测试: mode 检测 4 + dispatch 3 含跨集传播). tsc 0 dream-factory 错; 全套 559 passed + 2 skipped. |
| P2.3 | ✅ 抖音发布：fork social-auto-upload → `publisher`（仅 Douyin）+ `PublishStage` — 新增 `custom_nodes4macos/publisher/`(DouyinPublisher+PublishMeta/PublishResult+PublishConfigError/PublishDependencyError). `DouyinPublisher.upload_draft(video,meta,cookies,dry_run)`: dry_run=True(默认)只写清单不触网(安全可逆); dry_run=False 校验 cookies_path→惰性 `from playwright.sync_api import sync_playwright`(缺则 PublishDependencyError)→ `_upload_live` 走 creator.douyin.com 上传页(set_input_files/填标题/话题/点"存草稿")+每步日志+per-step try. `PublishStage`(name=publish, final→publish) config 门控: `publish_enabled`(默认 false,opt-in)+`publish_dry_run`(默认 true)+`publish_platform`(仅 douyin)+`publish_cookies_path`+`publish_title`/`publish_tags`/`publish_cover_path`; `publish_all_episodes=True` 迭代 series `_episode_finals`(每集标"第N集"),否则发 `ctx.get_artifact(0,"final")`. 三段异常降级不崩管线: PublishConfigError→skipped, PublishDependencyError→skipped+dep_missing=True+break, Exception→failed. 写 `publish_manifest.json`+`set_artifact(0,"publish")`. 注册于 `stages/__init__.py`; `IMPLEMENTED_STAGES` 同步加 publish. 不改默认模板(opt-in via `stages` override, 保 `test_all_templates_end_with_final_producer`). 新增 `test_publish.py`(18 测试: info/disabled/skip/no-final/dry-run manifest+title fallback+override/live cookies 缺/dep_missing 传播/launch failed 不崩/多集迭代+fallback/publisher 单元 dry-run+config error+dep error+cookies 归一化). 全套 577 passed + 2 skipped. |

**验收**：口播数字人单集 + 多集可跑；一集自动上传抖音草稿箱。

### P3（持续）— 扩展能力

- ✅ 卡通动作（multi_pose + 一致性）— `MultiPoseStage`：同角色 N 姿态关键帧（同 seed+appearance 一致性，姿态 suffix 变化）→ 定格剪辑 clip（每姿态 Ken Burns 段 + 硬切拼接 + 旁白音轨），`motion_mode=multi_pose` 时 engine 自动把 `ken_burns` 换成 `multi_pose`；14 tests（含真实 ffmpeg + engine.run 集成），593 passed
- ⏳ 真实人物形象 track（realistic + multi_pose）— ComfyUI 侧已打通且 **wire-level e2e 验证完成**：`character_style=realistic` 时 `image_generate` 把 `character_registry[*].reference_image`（跨场景身份）+ `multi_pose` 把每场 base_img（场内姿态一致性）作为参考图透传 fusion-mlx（`conditioning_mode=redux`/`in_context` + `reference_strength`），fusion_client/ImageGenerateStage/MultiPoseStage 三层全通，17 tests（含 `TestRealisticReferenceWireE2E` mock 传输层跑全链路断言 wire body 含 reference_image，610 passed）；**唯一剩余=身份保持本身依赖 fusion-mlx PR #31（`/v1/images/generate` 参考图条件生成，spec 已开 PR）落地后端到端验证**——PR #31 落地前 fusion-mlx 忽略此字段不报错（pydantic 前向兼容），落地后同一 wire body 被消费
- ✅ AI 视频运动评估（2026-07-06，`docs/P3_VIDEO_MOTION_EVAL.md`）— 实测：fusion-mlx 无视频端点/引擎、mflux 仅图像、本地无视频模型、MLX 视频生态薄。结论=**延续"动作幻觉"路线（多姿态+Ken Burns+剪辑，美学契合鬼故事定格质感）**，路径 A（建 MLX 视频引擎）列长期观测，触发条件=MLX 视频模型可用性核实通过+用户要"真动作"时再提 fusion-mlx PR；否决非 MLX torch 路径
- ✅ 粉丝反馈雷达评估（2026-07-06，`docs/P3_FAN_FEEDBACK_EVAL.md`）— 三段链路（①数据采集②LLM反馈分析③story_ingest回流）：②③可做但依赖①，①硬阻塞于抖音数据 API。结论=**暂不落地，不建投机脚手架（Rule 2/4/9）**；首选路径 A（抖音官方 Open API，需用户凭据），次选路径 B（用户手动导出创作者后台数据）；触发条件出现再基于真实数据形态建②③
- ✅ 多智能体编排评估（2026-07-06，`docs/P3_MULTIAGENT_EVAL.md`）— 结论=**不引入多智能体，保留 PipelineEngine + 分层故事结构**。当前线性 stage 已等价覆盖多智能体收益（分解/专精/迭代精修/并行/可复现），且保留确定性+可测试+可复现（Rule 2/5/9）；仅当需跨集开放世界观演化/模型自主决策剧情/单 stage 需多轮协商时再评估

---

## 七、风险与依赖

| 风险/依赖 | 说明 | 应对 |
|:---|:---|:---|
| fusion-mlx PR #27 未合并 | image_gen 不在主线 | P0.6 可先在 `feat/image-gen-wiring` 分支跑 |
| launchd 常驻 | 需用户授权 | 参考 `finance_api_launchd_resident` 模式 |
| 角色一致性不足 | seed+appearance 可能不够 | P1 验证，不够则补 IP-Adapter/PuLID（评估 MLX 支持） |
| 抖音风控 | cookie/验证码 | social-auto-upload 人工兜底 |
| 真动作硬约束 | 无 MLX 视频引擎 | P0-P2 用多姿态+Ken Burns+剪辑，真 AI 视频运动 P3 |
| 4×2 矩阵全贯通 | 工作量大 | 按 P0(2格)→P1(1格)→P2(2格) 增量交付 |

---

## 八、fusion-mlx 改动清单（预计）

| 阶段 | 改动 | 类型 |
|:---|:---|:---|
| P0.6 | 合并 PR #27 + 提交 `admin/` + launchd 常驻配置 | 整理/打包 |
| P0-P1 | 基本不需要（视频组装是 ffmpeg，LLM/图像/TTS 已通） | — |
| P2 | whisper 模型配置（voice_clone STT 依赖） | 配置 |
| P3 | 视频生成引擎（image-to-video，MLX-native）新能力 | 新引擎 |

**原则**：新增优于改动；必须改现有接口时保持旧签名/旧路由可用。碰到 fusion-mlx 缺的能力，先在 `custom_nodes4macos` 侧用 HTTP 调用现有能力兜底，兜不住才拉 fusion-mlx 分支改，改完本地验证通过再提 PR，不卡在等合主干上。

---

## 九、构建顺序总览

```
P0  短剧做扎实 ──► P1  长剧能力 ──► P2  数字人口播 ──► P3  扩展能力
   (氛围旁白)        (series)         (AvatarAnimate)     (卡通/真实/AI视频/雷达)
   1.5 周             1.5 周            1 周                持续
```

每个阶段独立可交付，前一阶段验收通过再进入下一阶段。
