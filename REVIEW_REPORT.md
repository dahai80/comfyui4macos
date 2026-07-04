# comfyui4macos 综合审视报告

> **项目**: 梦工厂 Dream Factory — MLX-native AI 内容生产管线
> **版本**: 1.0.0 (Beta)  **License**: MIT  **作者**: dahai80
> **审视日期**: 2026-07-04
> **审视范围**: 业务架构 / 技术架构 / 性能 / DFX / 代码本身 / 测试
> **代码规模**: 28 个 Python 文件，5244 LOC（业务），12 个测试文件，7 YAML 模板，6 ComfyUI 节点
> **审视方法**: 静态阅读全部源码 + 运行测试套件 + 架构推演

---

## 一、执行摘要

| 维度 | 评级 | 一句话结论 |
|------|------|------------|
| 业务架构 | ⭐⭐⭐⭐ | 7 内容类型 + 7 阶段管线 + 模板化，业务建模清晰，扩展点设计良好 |
| 技术架构 | ⭐⭐⭐⭐ | 顺序加载显存治理是亮点；双路径（MLX 原生/HTTP 回退）弹性强 |
| 性能 | ⭐⭐⭐ | 显存治理优秀；但 GPU 利用率、并行粒度、I/O 有明显改进空间 |
| DFX（可靠性/可观测/安全） | ⭐⭐⭐ | checkpoint 幂等扎实；可观测性薄、安全风险存在、错误处理不均衡 |
| 代码质量 | ⭐⭐⭐⭐ | 类型注解完整、日志规范、抽象分层清楚；存在重复与少量死代码 |
| 测试 | ⭐⭐⭐⭐ | 136 passed/3 skipped，单测覆盖面广；但缺集成测试与端到端验证 |

**总体判断**: 架构成熟度超出 v1.0 Beta 阶段的预期，**显存治理与幂等设计达到生产级水准**。主要风险集中在**可观测性、安全（subprocess/env）、集成测试缺失、GPU 利用率**四个方面。**未发现阻断性（P0）缺陷**。

---

## 二、业务架构审视

### 2.1 业务模型

项目定位为"**一键出片**"的本地 AI 内容工厂，覆盖 7 种内容形态：

| 内容类型 | 时长 | 关键差异 | 模板阶段链 |
|----------|------|----------|------------|
| short_drama 短剧 | 2-3min | 静态图+Ken Burns+TTS | prompt_expand→image→tts→ken_burns→assemble |
| ad_drama 广告剧 | 1-5min | 品牌植入 | （模板未读，链路同上推断） |
| puppet_show 木偶剧 | 5-15min | 角色一致性 | 同上 + consistency_check |
| medium_video 中短视频 | 30min | checkpoint/续跑+一致性 | 同上 + checkpoint_every_n_scenes=5 |
| series 连续剧 | 30×30min | PDF→30集 | story_ingest→prompt_expand→...→assemble |
| digital_human 数字人 | 可变 | TTS+口型同步 | prompt_expand→tts→digital_human_render→assemble |
| digital_human_live 数字人直播 | 实时 | 占位 | （模板存在，实现为占位） |

**评价**：
- ✅ **业务建模精准**：7 种内容形态覆盖了短剧/广告/连载/数字人四大主流形态，且区分了"静态图+推镜"与"数字人"两条技术路线
- ✅ **模板化零代码扩展**：README 承诺"添加新内容类型只需 YAML+prompt 文件，零 Python 代码"，代码中 `_load_templates` + `_merge_config` 确实兑现了这一承诺
- ✅ **故事管线分层**：story_ingest（章节拆分→分集大纲）→ prompt_expand（分集→分镜）的二级拆分对长篇连载合理
- ⚠️ **业务闭环不完整**：
  - `digital_human_live` 仅有 YAML 模板占位，无 stage 实现，README 也标注"placeholder for future livestream mode"
  - `digital_human` 的 lip_sync（Wav2Lip/SadTalker）显式 `raise NotImplementedError`，仅 fallback 静态合成
  - `ad_drama`、`puppet_show` 模板未在本次审视中读取，但节点 `dream_factory.py` 的 content_type 列表已声明，需确认模板 stage 链与 README 描述一致

### 2.2 业务流程闭环

```
用户输入(story_seed/story_file)
   ↓
PipelineEngine.run(content_type, **overrides)
   ↓
模板加载 → config 合并 → job_dir 创建 → checkpoint 初始化
   ↓
Stage 顺序执行（每 stage 完成即 checkpoint）
   ↓
final_video = ctx.get_artifact(0, "final")
   ↓
返回 PipelineResult(job_id, job_dir, final_video)
```

**评价**：
- ✅ **闭环完整**：从故事种子到最终 mp4，每阶段产物落盘，可断点续跑
- ✅ **resume 路径独立**：`_resume()` 单独处理，复用 checkpoint，不重跑已完成 stage
- ⚠️ **业务监控缺位**：没有任务级状态机（running/paused/failed/cancelled），仅有 `progress` dict 但未持久化到 checkpoint 的顶层结构，前端轮询依赖 ComfyUI 的 `/history` 而非自建状态

### 2.3 业务风险

| 风险 | 影响 | 等级 |
|------|------|------|
| 数字人lip_sync未实现 | 数字人内容类型仅能产出"静态头像+音频"的合成视频，与"口型同步"业务承诺有差距 | P1 |
| 直播占位 | digital_human_live 无任何实现，UI 已暴露选项 | P2 |
| LLM 输出 schema 不稳定 | `_parse_and_validate_raw` 对 bare list/非 dict 多次兜底，说明模型常不遵守 schema，业务质量受模型能力制约 | P1（产品级，非代码级） |

---

## 三、技术架构审视

### 3.1 分层架构

```
ComfyUI 节点层 (nodes/)        ← 6 个节点，ComfyUI 注册接口
   ↓ 调用
管线引擎层 (pipeline/engine)    ← 模板加载、stage 编排、resume
   ↓ 编排
Stage 层 (pipeline/stages/)     ← 7 个 stage，ABC 基类 + 自动注册
   ↓ 使用
模型治理层 (model_manager)      ← 顺序/常驻加载、显存预算、释放
   ↓ 调用
基础设施层 (fusion_client/ffmpeg_util)  ← HTTP 客户端、ffmpeg 封装
```

**评价**：
- ✅ **分层清晰**：节点层薄（仅参数适配）、引擎层纯编排、stage 层纯业务、基础设施层可复用
- ✅ **Stage 自动注册**：`register_stage` 装饰器 + `_auto_discover_stages` glob 导入，新增 stage 只需放文件并调用 register，符合"约定优于配置"
- ✅ **双路径设计**：每个 ML stage（image_generate/tts/prompt_expand/story_ingest）都有 `_mlx`（本地原生）与 `_http`（fusion-mlx 服务）双实现，`ImportError` 自动回退，弹性极强
- ⚠️ **节点与 stage 双轨重复**：`nodes/ken_burns.py` 与 `pipeline/stages/ken_burns.py` 是**两份独立实现**，`nodes/assemble.py` 与 `pipeline/stages/assemble.py` 同样。节点版供 ComfyUI 工作流单点调用，stage 版供管线编排，但 zoompan 构造、filter_complex 构造逻辑**完全重复**，维护时易漏改

### 3.2 ModelManager 显存治理（核心亮点）

```python
ModelManager(mode=SEQUENTIAL, memory_budget_gb=auto)
  ├── acquire(name) → _AcquireContext → _acquire_handle
  │     ├── 命中缓存(RESIDENT) → 返回
  │     ├── 超预算 → 释放常驻模型 → 仍超 → MemoryError
  │     └── 加载 → 计入 _current_usage
  └── release(name) → del + gc.collect + mx.clear_cache
```

**评价**：
- ✅ **顺序加载是正确决策**：Qwen3.5-9B(5.6G)+Flux(7.0G)+TTS(2.9G)=15.5G 同时驻留会击穿多数消费级 Mac，SEQUENTIAL 模式下峰值=7.0G（Flux），README 的核心承诺兑现
- ✅ **显存预算自动探测**：`_detect_memory_budget` 用 `sysctl hw.memsize` 取物理内存，`min(total*0.6, total-4)` 留 4G 给系统，下限 8G，符合 macOS 实际
- ✅ **acquire 上下文管理器**：`with mgr.acquire("llm") as handle` 在 SEQUENTIAL 模式 `__exit__` 自动 release，调用方无法泄漏
- ⚠️ **_current_usage 是估算值非实测**：用 `reg["memory_gb"]` 累加，不读 `mx.get_active_memory()`，若模型实际显存偏离标注（如不同量化版本），预算判断会失准
- ⚠️ **release 下溢保护粗糙**：`if self._current_usage >= reg["memory_gb"]` else 归零，多次 release 同一模型会让 usage 提前归零（虽然 `_loaded` 已 del 不会重复释放，但调用方直接调 `release` 不走 `__exit__` 时有风险）

### 3.3 CheckpointManager 幂等

```python
CheckpointManager(job_dir)
  ├── save(ctx) → 序列化 dataclass → _checkpoint.json
  └── load() → 反序列化 → 字段白名单过滤 → CheckpointData
```

**评价**：
- ✅ **stage 级幂等**：engine 在每 stage 完成后 `checkpoint.save(ctx)`，resume 时 `_skip_if_completed` 跳过
- ✅ **scene 级幂等**：image_generate/tts/ken_burns 内部每 N scenes 存一次（`checkpoint_every_n_scenes`），且 `has_artifact_on_disk` 检查产物文件存在+非空，双重幂等
- ✅ **腐败容忍**：`load()` 对 JSONDecodeError/TypeError/非 dict root 都返回 None 而非抛异常，字段白名单 `{k for k in raw if k in __dataclass_fields__}` 忽略未知字段
- ⚠️ **非原子写入**：`json.dump` 直接覆写 `_checkpoint.json`，若写入中途进程被杀（OOM/掉电），文件会损坏。生产级应写 `_checkpoint.json.tmp` 再 `os.replace` 原子替换
- ⚠️ **无版本号**：`CheckpointData` 无 schema_version 字段，未来字段变更后旧 checkpoint 反序列化可能静默丢失字段（虽有白名单兜底，但无迁移机制）

### 3.4 ComfyUI 集成边界

- ✅ **节点注册规范**：`NODE_CLASS_MAPPINGS` + `NODE_DISPLAY_NAME_MAPPINGS` + `WEB_DIRECTORY` 三件套齐全，`__init__.py` 用 try/except 逐节点注册，单节点失败不阻塞整体加载
- ✅ **web 扩展**：`dream_factory.js` 通过 `app.registerExtension` 注入按钮，`window.open` 打开 HTML 控制台，符合 ComfyUI 扩展规范
- ⚠️ **HTML 控制台是独立前端**：`dream_factory.html` 直接 fetch `/prompt` 提交工作流 JSON，绕过 ComfyUI 前端图编辑器，是简化方案但**无 CSRF 防护、无输入校验**（见安全）

### 3.5 技术债务

| 债务 | 位置 | 影响 |
|------|------|------|
| 节点/stage 双轨重复 | nodes/ken_burns vs stages/ken_burns, nodes/assemble vs stages/assemble | 维护成本×2，行为易漂移 |
| fusion_client 全局缓存 | `_models_cache`/`_default_model_cache` 模块级 dict | 进程内单例，无法按 job 隔离，TTL 60s 内模型列表变更不感知 |
| 硬编码模型路径 | model_manager.MODEL_REGISTRY | 切换模型需改源码，应外置到 YAML |
| FLUX_PIPELINE_DIR 默认 `~/claude-home/mlx-examples/flux` | model_manager._load_flux | 路径含 `claude-home` 暗示开发环境遗留，应改为更通用的默认 |

---

## 四、性能审视

### 4.1 显存与 GPU 利用

| 项 | 现状 | 评价 |
|----|------|------|
| 显存峰值 | 7.0G（仅 Flux 驻留） | ⭐⭐⭐⭐⭐ 优秀，消费级 Mac 可跑 |
| 模型加载次数 | SEQUENTIAL 模式每次 acquire 重新加载 | ⚠️ 同一 stage 内多次 acquire 同一模型会重复加载（如 image_generate 内已用 `with` 包住整个循环，正确；但跨 stage 若都需要 llm 会重复加载） |
| mx.clear_cache 调用 | release + 每 scene 生成后 | ✅ 合理，防止显存碎片 |
| MLX warmup | `_warmup_mlx` 仅 `mx.zeros(1)` | ⚠️ 仅初始化运行时，未预热实际模型，首次 stage 仍冷启动 |

### 4.2 并行性

| 项 | 现状 | 评价 |
|----|------|------|
| KenBurns 并行渲染 | `ThreadPoolExecutor(max_workers=ken_burns_workers)`，默认 2，series 模板 3 | ⭐⭐⭐ CPU 端 ffmpeg 并行，思路对 |
| 并行安全 | `ctx.set_artifact` 在主线程 as_completed 回调中调用，无锁 | ⚠️ PipelineContext.artifacts 是普通 dict，多线程并发写虽 GIL 下大概率安全，但非显式线程安全 |
| 错误处理 | `fut.exception()` 仅 log 不抛 | ⚠️ 并行中某 scene 失败被吞，最终 assemble 会发现 clip 缺失再跳过，错误传播链长 |
| GPU/CPU 重叠 | 无 | ⚠️ image_generate（GPU）与 ken_burns（CPU/VideoToolbox）是顺序的，未做流水线重叠 |
| TTS 并行 | 顺序逐 scene | ⚠️ TTS 是逐 scene 串行，但 TTS 模型在 GPU，本可与下一 scene 的 prompt 构造重叠 |

### 4.3 I/O 与存储

| 项 | 现状 | 评价 |
|----|------|------|
| 产物落盘 | PNG/WAV/MP4 全部写 `job_dir` | ✅ 便于 resume 与审计 |
| 临时文件 | ken_burns 用 `tempfile.mkdtemp` | ⚠️ 未清理，长期运行会堆积临时目录 |
| checkpoint 频率 | 每 stage + 每 N scenes | ✅ 平衡了持久化开销与恢复粒度 |
| 重复 ffprobe | assemble 中 `probe_duration`/`probe_has_audio` 对每个 clip 多次调用 | ⚠️ 每次 subprocess.run 启动 ffprobe 进程，n 个 clip × 2 probe = 2n 进程，可缓存 |

### 4.4 ffmpeg / VideoToolbox

- ✅ **VideoToolbox 探测缓存**：`_VT_CACHE` 全局缓存，避免每次重跑 `ffmpeg -encoders`
- ✅ **编码器降级**：有 VT 用 `h264_videotoolbox -q:v`，无则 `libx264 -preset ultrafast -crf 23`
- ⚠️ **VT 质量参数**：`-q:v 65` 对 h264_videotoolbox 偏高（VT 的 q:v 范围 1-100，越低越好），65 可能产出偏模糊；libx264 的 crf 23 是合理默认，两者质量档位不对等
- ⚠️ **timeout**：ken_burns 600s、assemble 900s，长视频（30 集×30min）的 assemble 可能超时

### 4.5 性能改进建议（优先级）

1. **P1 GPU/CPU 流水线**：image_generate 与 ken_burns 之间用队列衔接，前一个 scene 的 image 完成立即入队 ken_burns，GPU 不空等 CPU
2. **P2 模型常驻可选**：对 series 场景（30 集都用 llm），提供 RESIDENT 模式开关，避免每集重新加载 5.6G
3. **P2 TTS 并行**：TTS 单 scene 耗时短但 scene 多，可用 ThreadPool 并行合成多 scene 音频
4. **P3 ffprobe 缓存**：assemble 中缓存 clip 的 duration/has_audio
5. **P3 临时目录清理**：ken_burns 的 `tempfile.mkdtemp` 用 try/finally shutil.rmtree

---

## 五、DFX 审视（可靠性/可观测/安全/可维护）

### 5.1 可靠性

| 项 | 现状 | 评价 |
|----|------|------|
| 断点续跑 | stage + scene 双级 checkpoint | ⭐⭐⭐⭐⭐ |
| 网络重试 | fusion_client 5xx/429/408 重试 + 指数退避 | ⭐⭐⭐⭐ |
| 模型加载失败 | 抛 MemoryError/ImportError，engine 捕获存 checkpoint 后 re-raise | ✅ |
| ffmpeg 失败 | 捕获 returncode + stderr tail 800 字符 | ✅ |
| 原子写入 | checkpoint 直接覆写 | ⚠️ P2（见 3.3） |
| 部分失败处理 | ken_burns 并行 scene 失败仅 log | ⚠️ P2 |

### 5.2 可观测性（最大短板）

| 项 | 现状 | 评价 |
|----|------|------|
| 日志 | 全模块 `logging.getLogger`，结构化 `{asctime} [mod] {level} {msg}` | ⭐⭐⭐⭐ 基础扎实 |
| 日志级别控制 | `CUSTOM_NODES4MACOS_LOG_LEVEL` 环境变量 | ✅ |
| 指标 | **无** | ⚠️ 无 stage 耗时/显存/吞吐指标采集 |
| tracing | **无** | ⚠️ 无 job/stage/scene 三级 trace ID |
| 进度上报 | `ctx.update_progress` 写 dict，但 engine 不推送 | ⚠️ 前端靠 ComfyUI `/history` 轮询，与 ctx.progress 脱节 |
| 显存监控 | 仅 `_current_usage` 估算 | ⚠️ 不读 `mx.get_active_memory()` 实测 |
| 错误可定位性 | log 含 stage 名 + 异常 str，但无堆栈 | ⚠️ `logger.error("stage %s failed: %s", info.name, exc)` 未用 `exc_info=True`，丢失 traceback |

**改进建议**：
- P1 关键路径加 `logger.error(..., exc_info=True)` 保留堆栈
- P2 引入 stage 耗时 metric（`time.monotonic()` 包裹 stage.process）
- P2 engine 将 `ctx.progress` 写入 checkpoint 顶层，前端可读

### 5.3 安全

| 风险点 | 位置 | 等级 | 说明 |
|--------|------|------|------|
| subprocess 注入 | ffmpeg_util.run_ffmpeg | ⭐⭐⭐⭐ 低 | 命令用 list 形式 `subprocess.run(cmd, ...)`，非 shell，参数注入风险低；但 `label` 进 log 前未脱敏 |
| 任意文件读取 | story_ingest._read_file | ⭐⭐⭐ 中 | `story_file` 参数用户可控，可读任意路径（/etc/passwd 等），ComfyUI 本地单用户场景风险可接受，但若暴露网络则高危 |
| 任意文件写入 | 产物路径由 job_dir 派生 | ⭐⭐ 低 | job_dir 由 engine 控制，但 nodes/dream_factory 的 `config_overrides` JSON 可注入任意 config key |
| API key 泄露 | fusion_client 日志 | ⭐⭐⭐ 中 | `client init base_url=%s` 不打 api_key（✅），但 `chat` 的 messages 内容进 log，若用户在 prompt 中放敏感信息会落日志 |
| env 注入 | DEFAULT_* 环境变量 | ⭐⭐ 低 | `os.environ.get` 无类型校验，`FUSION_MLX_TIMEOUT` 传非数字会 float() 抛异常，但仅启动失败 |
| HTML 控制台无鉴权 | dream_factory.html | ⭐⭐⭐ 中 | fetch `/prompt` 无 CSRF token、无输入长度限制，story_seed 可注入超长文本耗尽 LLM |
| base64 解码 | fusion_client.generate_image | ⭐ 低 | `base64.b64decode` 对非标准输入宽松，但来源是受信服务响应 |

**改进建议**：
- P2 story_file 增加路径白名单或 sandbox 检查（限制在 `~/Documents` 或指定目录）
- P2 HTML 控制台对 story_seed 增加长度上限（如 10000 字符）
- P3 log 中对 prompt/audio_script 内容截断（如前 200 字符）

### 5.4 可维护性

| 项 | 现状 | 评价 |
|----|------|------|
| 类型注解 | 全量 `from __future__ import annotations` + 参数/返回类型 | ⭐⭐⭐⭐⭐ |
| 文档字符串 | 关键类有 docstring（如 `"""梦工厂 — 一键出片"""`），方法级偏少 | ⭐⭐⭐ |
| 命名一致性 | 中英混用：节点类 `FusionMLX*`，stage 类 `*Stage`，模块 `horror_tts` vs `tts_synthesize` | ⚠️ horror_tts 命名残留（早期为恐怖剧专用，现已通用） |
| 配置外置 | 模板 YAML ✅，但模型注册表硬编码 | ⚠️ 见 3.5 |
| 依赖管理 | pyproject.toml + requirements.txt 双份，且内容不一致（requirements 有 torch/scipy，pyproject 无） | ⚠️ P2 应统一 |

---

## 六、代码本身审视

### 6.1 优点

- ✅ **类型注解完整**：几乎全部函数有参数与返回类型，用了 `str | None` 现代 union 语法（Python 3.10+）
- ✅ **dataclass 用得到位**：StageInfo(frozen)、CheckpointData、PipelineResult，不可变与可变区分清楚
- ✅ **ABC 抽象正确**：Stage 抽象基类 + abstractmethod，_skip_if_completed 提供默认实现
- ✅ **异常分层**：FusionMLXError(RuntimeError) 自定义，HTTP/transport/解码错误都归一
- ✅ **日志规范**：每模块独立 logger，关键路径都有 info/warning/error，参数化 `%s` 而非 f-string（性能+安全）

### 6.2 代码问题清单

| # | 文件:行 | 问题 | 等级 |
|---|---------|------|------|
| C1 | model_manager.py:143 | `release` 下溢逻辑：`if usage >= mem: usage-=mem else: usage=0`，若先 release 一个未加载模型，usage 误归零 | P2 |
| C2 | engine.py:88-100 | `_merge_config` 中 `config.update(user_config)` 在最后，但 `config["overrides"]=dict(user_config)` 在 update 前，若 user 传 `overrides` key 会先被存再被覆盖 | P3 |
| C3 | checkpoint.py:43 | `config_overrides=ctx.config.get("overrides", dict(ctx.config))`，fallback 是整个 config（含 stages/templates），非真正 overrides，序列化体积大 | P2 |
| C4 | dream_factory.py:104 | `cp_path = f"{result.job_dir}/_checkpoint.json"` 用字符串拼接，应用 os.path.join | P3 |
| C5 | ken_burns.py(stage):84 | `_render_parallel` 中 `ctx.set_artifact` 在 as_completed 回调中调用，但 `ctx.update_progress` 也在同处，若 ThreadPool 内部异常，done 计数仍+1，进度虚高 | P3 |
| C6 | image_generate.py:189 | `_generate_http` 中 `images[0] if isinstance(images[0], bytes) else base64.b64decode(images[0])`，但 fusion_client.generate_image 已返回 bytes 列表，此处 base64 分支是死代码 | P3（死代码） |
| C7 | prompt_expand.py(node):116 | bare list 兜底 `parsed = {"scenes": parsed}`，但 stage 版 `_parse_and_validate_raw` 已有同样逻辑，重复 | P3 |
| C8 | horror_tts.py:14 | `_AUDIO_EXTS = ("wav",)` 单元素元组，`response_format` 参数 INPUT_TYPES 又写死 `(["wav"], ...)`，扩展性差 | P3 |
| C9 | fusion_client.py:213 | `_models_cache`/`_default_model_cache` 模块级全局，测试间可能相互污染（test_performance_and_fixes 未隔离） | P3 |
| C10 | digital_human_render.py:86 | fallback 路径 `cmd += ["-y", clip_path]`，但 ffmpeg_util.run_ffmpeg 内部已加 `-y`，重复 | P3 |
| C11 | story_ingest.py:161 | `chunk[:12000]` 硬截断章节文本，长章节内容丢失，应记录或分块 | P3 |
| C12 | model_manager.py:165 | `FLUX_PIPELINE_DIR` 默认 `~/claude-home/mlx-examples/flux`，开发环境路径泄露 | P3 |

### 6.3 重复代码

| 重复对 | 位置 | 处置建议 |
|--------|------|----------|
| `_build_zoompan` | nodes/ken_burns.py:47 vs stages/ken_burns.py:144 | 抽到 ffmpeg_util 或共享模块 |
| assemble filter_complex 构造 | nodes/assemble.py:90-160 vs stages/assemble.py:90-160 | 抽公共函数 |
| `_parse_json` codefence 剥离 | nodes/prompt_expand.py:27 vs stages/prompt_expand.py:250 vs story_ingest.py:244 | 三处重复，抽 util |
| `_output_directory` fallback | nodes/ken_burns, nodes/assemble, nodes/horror_tts | 三处相同，抽共享 |
| `_generate_http` 健康检查 | image_generate/tts/prompt_expand/stages 都有 `if not client.health(): raise` | 可在 fusion_client 内统一 |

### 6.4 测试覆盖审视

**运行结果**: `pytest custom_nodes4macos/tests/ -v` → **136 passed, 3 skipped, 0 failed**（8.5s）

| 测试文件 | 用例数 | 覆盖重点 | 评价 |
|----------|--------|----------|------|
| test_stages.py | 30 | zoompan 各 preset、各 stage info、process skip 逻辑、JSON 解析 | ⭐⭐⭐⭐ 单测扎实 |
| test_new_stages.py | 22 | story_ingest 章节拆分/PDF dispatch/episode 解析、digital_human fallback | ⭐⭐⭐⭐ |
| test_performance_and_fixes.py | 23 | context 竞态、checkpoint overrides、model_manager 顺序/常驻/预算、warmup、ffmpeg util | ⭐⭐⭐⭐ 生产硬化到位 |
| test_pipeline_engine.py | 5 | run/resume/list_jobs | ⭐⭐⭐ 基本路径覆盖 |
| test_production_hardening.py | 13 | 节点元数据、模板 YAML 校验、checkpoint 腐败容忍、429 重试 | ⭐⭐⭐⭐ |
| test_model_manager.py | 5 | acquire/release/常驻缓存/unknown | ⭐⭐⭐ |
| test_checkpoint.py | 6 | save/load/restore/腐败 | ⭐⭐⭐⭐ |
| test_ken_burns.py | 5 | 真实 ffmpeg 渲染各 preset+音频 | ⭐⭐⭐⭐ 集成级 |
| test_assemble.py | 7 | 真实 ffmpeg concat+bgm+fade | ⭐⭐⭐⭐ 集成级 |
| test_horror_tts.py | 7 | TTS 节点 offline+live | ⭐⭐⭐ |
| test_flux_image.py | 6 | 图像节点 offline+live | ⭐⭐⭐ |
| test_prompt_expand.py | 7 | prompt 节点 offline+live | ⭐⭐⭐ |

**测试优点**：
- ✅ **mock 与真实结合**：stage/engine/client 用 mock，ffmpeg 用真实进程（test_ken_burns/test_assemble 真渲染 mp4）
- ✅ **live marker 设计好**：`@pytest.mark.live` 标记需 fusion-mlx 服务的用例，无服务时自动 skip
- ✅ **边界覆盖**：空输入、腐败 JSON、未知字段、文件删除竞态都有

**测试缺口**：

| 缺口 | 等级 | 说明 |
|------|------|------|
| 端到端管线集成测试 | P1 | 无任何测试从 PipelineEngine.run 到 final_video 全链路跑通（即使 mock 模型） |
| web/dream_factory.html 测试 | P2 | 前端 0 测试，fetch/poll 逻辑未验证 |
| nodes/dream_factory.py 集成 | P2 | 仅 mock PipelineEngine 测元数据，未测真实 produce 全流程 |
| 并行渲染测试 | P2 | ken_burns `_render_parallel` 路径无测试（仅测了 sequential 的节点版） |
| 显存预算边界 | P2 | test 仅测 budget=5 时 llm(5.6) 失败，未测多模型叠加/释放回填 |
| resume 完整链路 | P1 | test_resume_restores_checkpoint 仅验证 stage 被跳过，未验证 artifacts/scenes 实际恢复后继续执行 |
| 跨 stage 数据流 | P1 | 无测试验证 stage A 输出（scenes/artifacts）被 stage B 正确消费 |
| MLX 原生路径 | P2 | 所有 MLX 路径因环境无 mlx 而走 ImportError 回退，原生代码路径 0 覆盖 |

---

## 七、缺陷分级汇总

### P0（阻断性）：无

### P1（应优先修复）

| ID | 维度 | 缺陷 | 建议修复 |
|----|------|------|----------|
| P1-1 | 测试 | 无端到端管线集成测试 | 新增 `test_e2e_pipeline.py`，mock 全部模型，验证 run→stages→final_video 全链路 + resume 续跑 |
| P1-2 | 测试 | 无跨 stage 数据流验证 | 新增测试：prompt_expand 产出 scenes → image_generate 消费 → 断言 artifact_path 衔接 |
| P1-3 | 性能 | GPU/CPU 无流水线重叠 | image_generate 与 ken_burns 间用队列衔接，scene 级流水线 |
| P1-4 | DFX | 错误日志丢失堆栈 | 关键路径 `logger.error(..., exc_info=True)` |
| P1-5 | 业务 | 数字人 lip_sync 未实现却已 UI 暴露 | UI 标注"实验性"或移除选项，避免用户预期落差 |

### P2（计划修复）

| ID | 维度 | 缺陷 |
|----|------|------|
| P2-1 | 可靠性 | checkpoint 非原子写入 |
| P2-2 | DFX | 无 stage 耗时/显存指标 |
| P2-3 | 安全 | story_file 任意路径读取 |
| P2-4 | 安全 | HTML 控制台无输入长度限制 |
| P2-5 | 代码 | model_manager release 下溢逻辑 |
| P2-6 | 代码 | checkpoint config_overrides 序列化整个 config |
| P2-7 | 依赖 | pyproject.toml 与 requirements.txt 不一致 |
| P2-8 | 重复 | 节点/stage 双轨实现重复 |
| P2-9 | 测试 | web 前端 0 测试 |
| P2-10 | 测试 | resume 完整续跑链路未验证 |

### P3（择机修复）

C1-C12 代码问题清单中的 P3 项 + 模型注册表硬编码 + horror_tts 命名遗留 + 临时目录未清理 + ffprobe 重复调用 + VT 质量参数档位。

---

## 八、改进建议路线图

### 短期（1-2 周）
1. 补端到端集成测试（P1-1, P1-2）——这是发布生产版的前置条件
2. 关键路径加 `exc_info=True`（P1-4）
3. 统一 pyproject.toml 与 requirements.txt（P2-7）
4. checkpoint 原子写入（P2-1）

### 中期（1 个月）
1. GPU/CPU 流水线（P1-3）
2. 引入 stage 耗时 metric + 显存实测（P2-2）
3. story_file 路径白名单 + HTML 输入限制（P2-3, P2-4）
4. 抽取节点/stage 共享代码，消除双轨（P2-8）

### 长期（季度）
1. 模型注册表外置 YAML + schema_version 迁移机制
2. 数字人 lip_sync 实现（Wav2Lip MLX port）
3. 可观测性体系：structured logging + metrics + trace ID
4. web 前端测试 + 鉴权

---

## 九、结论

**comfyui4macos 在 v1.0 Beta 阶段展现了超越预期的架构成熟度**。显存顺序加载治理、stage+scene 双级幂等 checkpoint、MLX 原生/HTTP 双路径回退、模板化零代码扩展——这四项设计是项目的**技术亮点**，达到了生产级水准。

主要风险不在代码正确性（136 测试全绿），而在**集成验证缺失、可观测性薄弱、安全边界未加固**。建议在宣称"生产可用"前，优先补齐端到端集成测试与可观测性，这两项是从 Beta 走向 GA 的关键门槛。

数字人 lip_sync 与直播功能的未实现，应在 UI 与文档中明确标注"实验性/规划中"，管理用户预期。

---

*报告生成于 2026-07-04，基于 commit 时的代码状态。审视者：AtomCode (GLM-5.2)。*
