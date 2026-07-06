# 超长连续剧（100+ 回）生成的技术挑战与优化策略

> 以《西游记》100 回为例，如何在有限资源下生成长篇剧集
> 同时保证故事完整性、角色一致性、内容不大量删减

---

## 一、资源消耗测算

### 1.1 100 回的总量

| 维度 | 单集（30min） | 100 回总和 |
|:-----|:------------:|:----------:|
| 场景数 | 150 | **15,000** |
| 总帧数 (24fps) | 43,200 | **4,320,000** |
| Flux 图片生成 | 150 | **15,000** |
| TTS 合成 | 150 段 | **15,000 段** |
| 口型动画 | 150 段 | **15,000 段** |
| 原始视频大小 | ~500MB | **~50GB** |

### 1.2 如果不做优化

```
Flux:       30s × 15,000 = 125 小时
TTS:         5s × 15,000 = 21 小时
AvatarAnim: 20s × 15,000 = 83 小时
ffmpeg:      5s × 15,000 = 21 小时
─────────────────────────────────
总计:                  250 小时 ≈ 10.4 天
峰值内存:              14GB+ (Flux + TTS + 多角色)
磁盘占用:              50GB+ (中间文件)
```

---

## 二、内存优化策略

### 2.1 模型懒加载 + 即时卸载（已实现需强化）

```
当前模式：
  LLM(5.6G) → 用完释放 → Flux(7.0G) → 用完释放 → TTS(2.9G)
  峰值: 7.0G ✅

需要强化：
  多角色场景下，需要同时加载多个数字人 → 引入"按需加载队列"

策略：
  ┌─ 每场景开始前: mx.clear_cache() + gc.collect()
  ├─ 模型用完即 del + 强制回收
  ├─ 角色图像预编码+编码器卸载（已有）→ 扩展到 TTS
  └─ 场景渲染完: 逐帧释放 frame 内存
```

### 2.2 流式处理（关键）

**不把所有场景的中间文件保留到最终合成时**

```python
def process_series_streaming():
    """
    流式处理：每渲染完一个场景，立即追加到输出视频
    不保留中间文件，只保留最终 concat 列表
    """
    concat_list = []
    for scene_idx in range(15000):
        # 1. 渲染场景 → clip.mp4
        clip = render_single_scene(scene_idx)
        
        # 2. 立即追加到流式输出
        append_to_streaming_output(clip)
        
        # 3. 删除中间文件（只保留 clip.mp4）
        cleanup_temp_files(scene_idx)
        
        # 4. 记录到 concat list
        concat_list.append(clip)
        
        # 5. 强制内存回收
        import gc; gc.collect()
        import mlx.core as mx; mx.clear_cache()
    
    # 最终通过 concat 合并所有 clip
    return ffmpeg_concat(concat_list)
```

**磁盘占用从 50GB 降到 5GB**（只保留最终 clip，不保留中间 PNG/音频）

### 2.3 推理缓存

```python
# TTS 缓存：相同文本不重复合成
tts_cache = {}
for scene in scenes:
    text = scene["audio_script"]
    if text not in tts_cache:
        tts_cache[text] = tts_synthesize(text, voice_id)
    # 否则直接用缓存结果

# Flux 缓存：相同视觉提示不重复生成
flux_cache = {}
for scene in scenes:
    prompt = build_prompt(scene["visual_prompt"])
    if prompt not in flux_cache:
        flux_cache[prompt] = flux_generate(prompt)
    # 否则直接用缓存结果
```

**效果：重复场景/相似场景直接复用，TTS 省 60%，Flux 省 40%**

---

## 三、算力优化策略

### 3.1 并行流水线（核心）

```
串行（当前）:
  [场景1渲染] → [场景2渲染] → [场景3渲染] → ...
  单个核心，总时间 = 各场景之和

并行流水线:
  Flux:     [场景1] → [场景2] → [场景3] → ... (单线程)
  TTS:               [场景1] → [场景2] → ... (单线程)
  AvatarAnimate:               [场景1] → ... (4线程已实现)
  ─────────────────────────────────────────
  流水线并行，总时间 = 各阶段最慢者
```

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

# 3 级流水线
async def pipeline_3stage():
    flux_queue = asyncio.Queue(5)     # Flux 最多预生成 5 张
    tts_queue = asyncio.Queue(10)     # TTS 最多预合成 10 段
    render_queue = asyncio.Queue(3)   # 渲染最多排队 3 个
    
    # 线程1: Flux 生成场景背景（GPU 密集型，单线程）
    # 线程2: TTS 合成（GPU 密集型，单线程）
    # 线程3-6: AvatarAnimate（CPU 密集型，4 线程已有）
    
    # 效果：总时间 ≈ 最慢阶段 × 场景数
    # 而不是 各阶段之和 × 场景数
```

**效果：总时间从 250 小时降到约 83 小时（AvatarAnimate 阶段）**

### 3.2 批量化

```python
# Flux 批量化：合并相同/相似 prompt
# 例：同一角色在相似场景中，prompt 只有场景名不同
base_prompt = "孙悟空，身穿金甲，头戴紫金冠"
scenes = {
    "花果山": f"{base_prompt}，在花果山水帘洞前",
    "天庭":   f"{base_prompt}，在天庭凌霄殿上",
    "火焰山": f"{base_prompt}，在火焰山前",
}
# 可以一次性生成 3 张同角色不同场景的图
# Flux 的 VAE encode 可以 batch 处理（但 transformer 需逐张）
```

**效果：角色图像生成省 20-30%**

### 3.3 混合精度与量化

```python
# 当前
Flux-1.lite-8B-MLX-Q4  (4bit 量化，7.0GB)
Qwen3.5-9B-4bit        (4bit 量化，5.6GB)
Qwen3-TTS-12Hz-1.7B    (8bit 量化，2.9GB)

# 进一步优化
# 1. TTS 模型降到 4bit（实验性，可省 1GB）
# 2. Flux 推理时用 float16（MLX 默认），减少显存带宽
# 3. 所有模型统一在 GPU 上，避免 CPU-GPU 传输
```

---

## 四、内容一致性策略（故事不删减）

### 4.1 分层故事结构

```
原始小说（100 回）
       ↓
┌─────────────────────────────────────┐
│ 层1: 故事大纲（LLM 摘要）             │
│  → 每回 200 字摘要                    │
│  → 保留所有关键情节节点                │
│  → 识别角色出场/退场                   │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 层2: 每回场景拆分（LLM 扩写）          │
│  → 每回拆成 8-12 个场景               │
│  → 每个场景 10-15 秒                  │
│  → 标注角色、对话、动作                │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 层3: 场景级渲染（Pipeline）           │
│  → 逐场景生成                        │
│  → 流式输出                          │
└─────────────────────────────────────┘
```

**每回 200 字 → 8 场景 × 15 秒 = 2 分钟**
**100 回 = 200 分钟 ≈ 3.3 小时**（比 30 分钟更合理）

### 4.2 删减控制策略

| 策略 | 说明 | 效果 |
|:-----|:-----|:----:|
| 情节保留率 | LLM 摘要时指定保留所有关键节点 | 不删主线 |
| 删除标记 | LLM 标注"可删"和"必留"段落 | 只删次要描写 |
| 自动压缩 | 长段落自动缩短（非删除） | 保留所有情节 |
| 用户分级 | 提供 3 档时长选项（完整/标准/精简） | 用户自己选择 |

```python
# 删减控制参数
compression_config = {
    "mode": "standard",      # full | standard | compact
    "retain_key_plots": True, # 保留所有关键情节
    "min_scene_duration": 8,  # 最短场景 8 秒
    "max_scene_duration": 20, # 最长场景 20 秒
    "description_ratio": 0.6, # 场景描述压缩到 60%
    "dialogue_keep_all": True, # 对话全部保留
}
```

### 4.3 跨回一致性

```python
# 跨回角色注册表（不仅仅是跨场景）
series_registry = {
    "title": "西游记",
    "characters": {
        "孙悟空": {
            "avatar_id": "avatar_sun_wukong",
            "voice_id": "voice_sun_wukong",
            "appearance": "孙悟空，身穿金甲，头戴紫金冠，手持金箍棒",
            "seed": 10001,        # 固定 Flux 种子
            "style_preset": "神话写实",
        },
        "唐僧": { ... },
        "猪八戒": { ... },
        "沙僧": { ... },
    },
    "continuity": {
        "current_episode": 1,
        "completed_episodes": [],
        "plot_state": {},        # 跨回情节状态
    }
}
```

**关键：每回开始前加载上一回的 registry，确保角色不变**

### 4.4 内容审核门控

```
生成流程中加入自动内容检查：
  每场景生成后 → 自动检查：
    1. 角色外貌是否与 registry 一致
    2. 对话是否与剧情相关
    3. 场景是否合理
  如果有问题 → 重新生成该场景
  最多重试 3 次
```

---

## 五、存储优化

### 5.1 三级存储策略

| 级别 | 内容 | 保留时间 | 大小 |
|:-----|:-----|:--------:|:----:|
| L1 热 | 当前回所有文件 | 生成期间 | 2GB |
| L2 温 | 已完成的回 (clip.mp4) | 永久 | 500MB/回 |
| L3 冷 | 最终拼接视频 | 永久 | 5GB/100回 |

```python
# 每回完成后：
# 1. 删除该回的中间文件（PNG/WAV/temp clips）
# 2. 只保留最终 clip.mp4
# 3. 更新 checkpoint（记录已完成回数）

def cleanup_episode(episode_dir):
    """每回渲染完后清理中间文件"""
    import shutil
    # 保留: scene_*_clip.mp4
    # 删除: scene_*_image.png, scene_*_audio.wav, _avatar/, tmp_*
    for f in os.listdir(episode_dir):
        if f.endswith("_image.png") or f.endswith("_audio.wav"):
            os.remove(os.path.join(episode_dir, f))
        if f.startswith("_tmp_") or f == "_avatar":
            shutil.rmtree(os.path.join(episode_dir, f), ignore_errors=True)
```

### 5.2 检查点策略

```
┌─ 场景级 checkpoint（已有） ─ 每 10 场景
├─ 回级 checkpoint（新增）   ─ 每完成一回
├─ 全局 checkpoint（新增）   ─ 每完成 10 回
└─ 支持从任意回续跑
```

---

## 六、《西游记》100 回专项测算

### 6.1 参数配置

| 参数 | 值 |
|:-----|:---:|
| 总回数 | 100 |
| 每回场景数 | 8-12 |
| 总场景数 | ~1,000 |
| 每场景时长 | 10-15 秒 |
| 单回时长 | 2-3 分钟 |
| 总时长 | ~200-300 分钟（3.3-5 小时） |
| 首次主角 | 孙悟空（贯穿全程） |
| 主要配角 | 唐僧、猪八戒、沙僧、白龙马 |
| 客串角色 | 如来、观音、妖怪等（每回不同） |

### 6.2 渲染时间估算

| 阶段 | 单场景 | 1,000 场景 | 优化后 |
|:-----|:------:|:----------:|:------:|
| Flux 场景背景 | 30s | 500min | **300min**（缓存复用） |
| Flux 角色图像 | 30s | 500min | **200min**（角色 seed 固定） |
| TTS 合成 | 5s | 83min | **33min**（文本去重缓存） |
| AvatarAnimate | 20s | 333min | **333min**（单线程瓶颈） |
| ffmpeg 合成 | 5s | 83min | **83min** |
| **合计** | **90s** | **1,500min** | **~950min (16h)** |

### 6.3 内存峰值

| 阶段 | 峰值内存 | 说明 |
|:-----|:--------:|:-----|
| LLM (摘要) | 5.6G | 每回一次，释放 |
| Flux (场景+角色) | 7.0G | 逐张生成，立即释放 |
| TTS | 2.9G | 每个场景一次 |
| AvatarAnimate | 2.0G | 多角色×2→4G（峰值） |
| 系统其他 | 2G | — |
| **峰值总计** | **~10G** | M 芯片可以承受 |

### 6.4 磁盘占用

| 项目 | 优化前 | 优化后 |
|:-----|:------:|:------:|
| Flux 中间图 | ~50G | 0（流式处理） |
| TTS 音频 | ~10G | ~2G（缓存去重） |
| 口型动画中间帧 | ~100G | 0（pipe 模式已有） |
| 最终 clip | ~5G | ~5G |
| **总计** | **~165G** | **~7G** |

---

## 七、推荐实施策略

### 7.1 路线图

```
第一步（1周）：改造现有 Pipeline 支持流式处理 + 内存优化
  → 验证：50 场景长剧跑通，峰值内存 < 12G

第二步（1周）：并行流水线 + 缓存系统
  → 验证：渲染速度提升 2x，磁盘占用降 80%

第三步（1周）：100 回超长剧 Pipeline + 检查点系统
  → 验证：西游记 10 回试跑

第四步（1周）：内容一致性 + 删减控制
  → 验证：100 回全跑完，情节保留率 > 95%
```

### 7.2 最低硬件要求

```
推荐: M5 Max 128GB    → 全程流畅
最低: M2 Pro 32GB     → 峰值压力大（需更激进的内存管理）
不推荐: 16GB 以下     → 频繁 swap，不可用
```

### 7.3 核心代码改动量

| 模块 | 改动量 | 说明 |
|:-----|:------|:-----|
| PipelineEngine | ~200行 | 流式模式 + 并行流水线 |
| AvatarAnimateStage | ~50行 | 多角色支持 |
| TTSSynthesizeStage | ~100行 | 缓存 + 批量化 |
| ImageGenerateStage | ~100行 | 缓存 + 角色 seed 固定 |
| AssembleStage | ~50行 | 流式 concat |
| checkpoint.py | ~50行 | 回级 checkpoint |
| 前端 | ~200行 | 超长剧配置 UI |
| **合计** | **~750行** | |

---

## 八、总结

```
超长剧（100 回）可行性的三大支柱：

1. 流式处理 → 不保留中间文件 → 磁盘从 165G 降到 7G
2. 并行流水线 → 3 级流水线 + 4 线程渲染 → 时间从 250h 降到 16h
3. 分层故事结构 → 每回 200 字摘要 → 100% 情节保留

在 M5 Max 128GB 上，100 回西游记约 16 小时可渲染完成。
在 M2 Pro 32GB 上，约 30 小时，峰值内存 < 10G。
```
