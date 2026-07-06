from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.prompt_expand")

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_CN_EN_KEYWORDS = {
    "破庙": "abandoned temple", "寺庙": "temple", "道观": "taoist temple",
    "木门": "wooden door", "大门": "door", "窗户": "window",
    "油灯": "oil lamp", "灯笼": "lantern", "蜡烛": "candle",
    "铜铃": "bronze bell", "铃铛": "bell",
    "香炉": "incense burner", "符咒": "talisman", "朱砂": "cinnabar",
    "纸人": "paper figure", "纸扎": "paper effigy",
    "白衣": "white dress", "老人": "old man", "女子": "woman",
    "少女": "young woman", "孩童": "child", "和尚": "monk",
    "道士": "daoist priest", "棺材": "coffin", "坟墓": "grave",
    "墓碑": "tombstone", "石桥": "stone bridge", "月光": "moonlight",
    "鲜血": "blood", "黑影": "dark shadow", "木鱼": "wooden fish",
    "经文": "sutra", "叹息": "sigh", "哭声": "crying", "笑声": "laughing",
    "推门": "pushing door", "回头": "turning back",
}


class PromptExpandStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="prompt_expand",
            description="故事种子 → 结构化分镜 JSON (LLM 驱动)",
            model_requirements=["llm"],
            memory_estimate_gb=5.6,
            input_kinds=["text"],
            output_kinds=["scenes"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        story_seed = ctx.config.get("story_seed", "")
        episodes = ctx.config.get("episodes", [])

        if episodes and isinstance(episodes, list) and len(episodes) > 0:
            self._process_episodes(ctx, model_manager, episodes)
            return

        if not story_seed or not story_seed.strip():
            raise ValueError("prompt_expand: story_seed is empty")

        episode_title = ctx.config.get("episode_title", "")
        scene_count = ctx.config.get("scene_count", 8)
        style_preset = ctx.config.get("style_preset", "")
        temperature = ctx.config.get("prompt_expand_temperature", 0.75)

        style_presets = ctx.config.get("style_presets", {})
        if style_preset and style_preset in style_presets:
            style_text = style_presets[style_preset]
        elif style_preset:
            style_text = style_preset
        else:
            first_key = next(iter(style_presets), "")
            style_text = style_presets.get(first_key, "")

        system_prompt_file = (
            ctx.config.get("system_prompt_file")
            or ctx.config.get("system_prompt")
            or "horror_director.md"
        )
        system_prompt = self._load_system_prompt(system_prompt_file)
        user_msg = self._build_user_message(
            story_seed, episode_title, scene_count, style_preset, style_text,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        parsed = PromptExpandStage._generate_with_retry(
            ctx, model_manager, messages, temperature,
            ep_idx=0, ep_title=episode_title or story_seed,
            ep_synopsis="", ep_key_scenes=[], ep_cliffhanger="",
            scene_count=scene_count,
        )
        scenes = parsed.get("scenes", [])

        if "global_style" in parsed and "global_style" not in ctx.config:
            ctx.config["global_style"] = parsed["global_style"]
            logger.info("global_style from LLM: %s", parsed["global_style"])

        self._merge_character_registry(ctx, parsed)
        self._enforce_chinese_faces(ctx)
        self._reinforce_visual_audio_correlation(scenes)

        ctx.scenes = scenes
        ctx.update_progress("prompt_expand", 1, 1)
        logger.info("prompt_expand done scenes=%d", len(scenes))

    def _process_episodes(self, ctx, model_manager, episodes) -> None:
        scene_count = ctx.config.get("scene_count", 8)
        style_preset = ctx.config.get("style_preset", "")
        temperature = ctx.config.get("prompt_expand_temperature", 0.75)

        style_presets = ctx.config.get("style_presets", {})
        if style_preset and style_preset in style_presets:
            style_text = style_presets[style_preset]
        elif style_preset:
            style_text = style_preset
        else:
            first_key = next(iter(style_presets), "")
            style_text = style_presets.get(first_key, "")

        system_prompt_file = (
            ctx.config.get("system_prompt_file")
            or ctx.config.get("system_prompt")
            or "series_director.md"
        )
        system_prompt = self._load_system_prompt(system_prompt_file)
        system_prompt = (
            "直接输出单个合法JSON对象，不要输出任何思考过程，不要输出任何解释文字，不要使用markdown代码块。"
            "JSON第一个字符必须是{，最后一个字符必须是}。\n"
            + system_prompt
        )

        all_scenes = []
        global_scene_offset = 0
        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)

        for ep_idx, episode in enumerate(episodes):
            ep_title = episode.get("title", f"第{ep_idx + 1}集")
            ep_synopsis = episode.get("synopsis", "")
            ep_key_scenes = episode.get("key_scenes", [])
            ep_cliffhanger = episode.get("cliffhanger", "")

            ep_seed = (
                f"【{ep_title}】\n"
                f"剧情概要：{ep_synopsis}\n"
            )
            if ep_key_scenes:
                ep_seed += f"关键场景：{', '.join(str(s) for s in ep_key_scenes)}\n"
            if ep_cliffhanger:
                ep_seed += f"悬念结尾：{ep_cliffhanger}\n"

            existing_registry = ctx.config.get("character_registry", [])
            user_msg = self._build_user_message(
                ep_seed, ep_title, scene_count, style_preset, style_text,
                character_registry=existing_registry,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]

            parsed = PromptExpandStage._generate_with_retry(
                ctx, model_manager, messages, temperature,
                ep_idx, ep_title, ep_synopsis, ep_key_scenes, ep_cliffhanger,
                scene_count,
            )
            scenes = parsed.get("scenes", [])
            for scene in scenes:
                scene["episode_id"] = episode.get("episode_id", ep_idx + 1)
                scene["episode_title"] = ep_title
            global_scene_offset = self._renumber_scenes(
                scenes, global_scene_offset,
            )

            if "global_style" in parsed and "global_style" not in ctx.config:
                ctx.config["global_style"] = parsed["global_style"]
                logger.info("global_style from LLM: %s", parsed["global_style"])

            self._merge_character_registry(ctx, parsed)

            existing_registry = ctx.config.get("character_registry", [])
            if existing_registry:
                ctx.config["character_registry"] = existing_registry

            self._reinforce_visual_audio_correlation(scenes)
            all_scenes.extend(scenes)
            ctx.scenes = all_scenes
            ctx.update_progress("prompt_expand", ep_idx + 1, len(episodes))
            logger.info("prompt_expand episode %d/%d scenes=%d", ep_idx + 1, len(episodes), len(scenes))

            if ctx.should_checkpoint_scene(ep_idx + 1):
                checkpoint.save(ctx)

        logger.info("prompt_expand all episodes done total_scenes=%d", len(all_scenes))
        self._enforce_chinese_faces(ctx)

    @staticmethod
    def _parse_and_validate(content: str) -> list[dict]:
        parsed = PromptExpandStage._parse_and_validate_raw(content)
        return parsed.get("scenes", [])

    @staticmethod
    def _parse_and_validate_raw(content: str) -> dict:
        parsed = PromptExpandStage._parse_json(content)
        if isinstance(parsed, list):
            logger.warning("model returned bare list, wrapping as {scenes: [...]}")
            parsed = {"scenes": parsed}
        if not isinstance(parsed, dict):
            raise RuntimeError(f"prompt_expand: output is not JSON object: {type(parsed).__name__}")

        scenes = parsed.get("scenes", [])
        if not scenes:
            raise RuntimeError("prompt_expand: no scenes in output")

        for i, scene in enumerate(scenes):
            if "scene_id" not in scene:
                scene["scene_id"] = i + 1

        return parsed

    @staticmethod
    def _generate_with_retry(ctx, model_manager, messages, temperature,
                             ep_idx, ep_title, ep_synopsis, ep_key_scenes,
                             ep_cliffhanger, scene_count) -> dict:
        parsed = None
        last_err = None
        for attempt in range(1, 4):
            try:
                with model_manager.acquire("llm") as handle:
                    content = PromptExpandStage._generate(handle, messages, temperature)
            except Exception as e:
                last_err = e
                logger.warning("prompt_expand ep%d generate failed attempt=%d: %s", ep_idx+1, attempt, e)
                continue
            raw_llm_path = os.path.join(ctx.job_dir, f"_prompt_expand_ep{ep_idx+1}_raw.txt")
            with open(raw_llm_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("prompt_expand ep%d raw output: %s (%d chars) attempt=%d", ep_idx+1, raw_llm_path, len(content), attempt)
            try:
                parsed = PromptExpandStage._parse_and_validate_raw(content)
                break
            except Exception as e:
                last_err = e
                logger.warning("prompt_expand ep%d parse failed attempt=%d: %s", ep_idx+1, attempt, e)
        if parsed is None:
            logger.warning(
                "prompt_expand ep%d: LLM failed 3 attempts (last_err=%s), using deterministic 西游记 fallback",
                ep_idx+1, last_err,
            )
            parsed = PromptExpandStage._fallback_scenes(
                ep_title, ep_synopsis, ep_key_scenes, ep_cliffhanger,
                scene_count, ep_idx, ctx,
            )
            fb_path = os.path.join(ctx.job_dir, f"_prompt_expand_ep{ep_idx+1}_fallback.json")
            with open(fb_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            logger.info("prompt_expand ep%d fallback saved: %s (%d scenes)", ep_idx+1, fb_path, len(parsed.get("scenes", [])))
        return parsed

    @staticmethod
    def _renumber_scenes(scenes: list[dict], offset: int) -> int:
        for i, scene in enumerate(scenes):
            scene["scene_id"] = offset + i + 1
        return offset + len(scenes)

    @staticmethod
    def _merge_character_registry(ctx, parsed: dict) -> None:
        char_reg = parsed.get("character_registry") or parsed.get("character_descriptions")
        if not char_reg:
            return
        existing = ctx.config.get("character_registry", [])
        existing_names = {c.get("name") for c in existing}
        for c in char_reg:
            name = c.get("name")
            if name and name not in existing_names:
                existing.append(c)
                existing_names.add(name)
            elif name and name in existing_names:
                for idx, ec in enumerate(existing):
                    if ec.get("name") == name:
                        if "appearance" not in ec and "appearance" in c:
                            existing[idx]["appearance"] = c["appearance"]
                        if "voice" not in ec and "voice" in c:
                            existing[idx]["voice"] = c["voice"]
                        break
        ctx.config["character_registry"] = existing
        logger.info("character_registry merged: %d characters", len(existing))

    @staticmethod
    def _enforce_chinese_faces(ctx) -> None:
        content_type = ctx.config.get("content_type", "")
        chinese_types = {"short_drama", "series", "medium_video", "puppet_show", "ad_drama"}
        if content_type not in chinese_types:
            return
        narrator_names = {"旁白", "narrator", "画外音", "叙述者"}
        char_reg = ctx.config.get("character_registry", [])
        enforced = 0
        for c in char_reg:
            name = c.get("name", "")
            app = c.get("appearance", "")
            if app:
                if "chinese" not in app.lower() and "east asian" not in app.lower():
                    c["appearance"] = f"Chinese face, East Asian features, {app}"
                    enforced += 1
            elif name and name not in narrator_names:
                c["appearance"] = "Chinese face, East Asian features"
                enforced += 1
        if enforced:
            logger.info("enforced Chinese face default for %d characters", enforced)
        gs = ctx.config.get("global_style", "")
        if gs and "east asian" not in gs.lower() and "chinese" not in gs.lower():
            ctx.config["global_style"] = f"{gs}, East Asian people by default"
            logger.info("enforced East Asian face default in global_style")

    @staticmethod
    def _reinforce_visual_audio_correlation(scenes: list[dict]) -> None:
        if not scenes:
            return
        sorted_keys = sorted(_CN_EN_KEYWORDS.keys(), key=len, reverse=True)
        reinforced = 0
        for scene in scenes:
            audio = scene.get("audio_script", "") or ""
            visual = (scene.get("visual_prompt", "") or "").lower()
            if not audio or not visual:
                continue
            present_cn = []
            for k in sorted_keys:
                if k in audio:
                    en = _CN_EN_KEYWORDS[k]
                    if en.lower() not in visual:
                        present_cn.append(en)
            if not present_cn:
                continue
            seen = set()
            uniq = []
            for e in present_cn:
                if e not in seen:
                    seen.add(e)
                    uniq.append(e)
            uniq = uniq[:6]
            scene["visual_prompt"] = f"{scene['visual_prompt']}, {', '.join(uniq)}"
            reinforced += 1
            logger.info(
                "visual-audio reinforcement scene %s: appended %s",
                scene.get("scene_id"), ", ".join(uniq),
            )
        if reinforced:
            logger.info(
                "visual-audio correlation reinforced for %d/%d scenes",
                reinforced, len(scenes),
            )

    @staticmethod
    def _load_system_prompt(filename: str) -> str:
        if os.path.isfile(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        path = _PROMPT_DIR / filename
        if not path.exists():
            logger.warning("system prompt missing: %s, using fallback", path)
            return "你是一位编剧，将故事种子扩展为分镜脚本，只输出 JSON。"
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _build_user_message(
        story_seed: str,
        episode_title: str,
        scene_count: int,
        style_preset: str,
        style_text: str,
        character_registry: list | None = None,
    ) -> str:
        msg = (
            f"故事种子：{story_seed.strip()}\n"
            f"剧集标题：{episode_title.strip() or '（待定）'}\n"
            f"目标分镜数：{scene_count}\n"
            f"画风预设：{style_preset}（{style_text}）\n"
        )
        if character_registry:
            reg_lines = []
            for c in character_registry:
                name = c.get("name", "?")
                appearance = c.get("appearance", "")
                voice = c.get("voice", "")
                line = f"  - {name}: appearance=\"{appearance}\""
                if voice:
                    line += f", voice=\"{voice}\""
                reg_lines.append(line)
            msg += (
                f"已有角色注册表（必须严格沿用，不得修改外观描述）：\n"
                + "\n".join(reg_lines) + "\n"
            )
        msg += (
            f"请严格按 schema 输出 JSON，分镜数必须等于 {scene_count}。"
            "思考结束后只输出JSON。"
        )
        return msg

    @staticmethod
    def _generate(handle, messages: list[dict], temperature: float) -> str:
        return PromptExpandStage._generate_http(handle, messages, temperature)

    @staticmethod
    def _fallback_scenes(ep_title, ep_synopsis, ep_key_scenes, ep_cliffhanger,
                         scene_count, ep_idx, ctx) -> dict:
        logger.warning("prompt_expand fallback: ep%d %s scenes=%d", ep_idx+1, ep_title, scene_count)
        storyboards = PromptExpandStage._storyboards()
        board = storyboards[ep_idx % len(storyboards)]
        scenes = []
        n = min(scene_count, len(board))
        for i in range(n):
            tpl = board[i]
            scenes.append({
                "scene_id": i + 1,
                "visual_prompt": tpl["visual_prompt"],
                "audio_script": tpl["audio_script"],
                "sound_effect": tpl["sound_effect"],
                "characters": tpl["characters"],
                "duration_seconds": 90,
            })
        while len(scenes) < scene_count:
            tpl = board[len(scenes) % len(board)]
            scenes.append({
                "scene_id": len(scenes) + 1,
                "visual_prompt": tpl["visual_prompt"],
                "audio_script": tpl["audio_script"],
                "sound_effect": tpl["sound_effect"],
                "characters": tpl["characters"],
                "duration_seconds": 90,
            })
        return {
            "story_title": ep_title,
            "global_style": "中国古典神话风格，工笔重彩与水墨写意结合，色彩饱和而庄重，光影戏剧化",
            "character_registry": [
                {"name": "孙悟空", "description": "石猴化形，金睛火眼，雷公嘴，孤拐面，身穿虎皮裙，手持金箍棒"},
                {"name": "菩提祖师", "description": "仙风道骨，白须长袍，手持拂尘，神态慈悲而威严"},
                {"name": "群猴", "description": "花果山猕猴众，形态各异，活泼顽皮"},
            ],
            "scenes": scenes,
        }

    @staticmethod
    def _storyboards() -> list[list[dict]]:
        ep1 = [
            {"visual_prompt": "Chinese mythological chaos cosmos, swirling primal energy, golden light breaking through darkness, ancient epic painting style, highly detailed, cinematic", "audio_script": "混沌未分天地乱，茫茫渺渺无人见。自从盘古破鸿蒙，开辟从兹清浊辨。在那天地初开的浩渺虚空之中，一股灵气凝结为石，静卧于花果山巅，等待千年一遇的机缘。", "sound_effect": "低沉的混沌风声，远处隐约雷鸣", "characters": ["旁白"]},
            {"visual_prompt": "Mount Huaguo Flower Fruit Mountain, a giant immortal stone on the peak glowing golden, mist and clouds, traditional Chinese landscape painting, majestic", "audio_script": "东胜神洲傲来国海中有一座花果山，山上有一块仙石。这仙石自开辟以来受天真地秀，日精月华，内育仙胞。一日忽然迸裂，产一石卵，化作一个石猴，五官俱备，四肢皆全。", "sound_effect": "山风呼啸，石裂之声", "characters": ["旁白"]},
            {"visual_prompt": "A stone monkey bursting from a giant glowing egg-shaped stone, golden light rays, sparks, epic moment, Chinese mythological art", "audio_script": "只听得一声巨响，那仙石迸裂开来，一个石猴从中跃出。他先是拜了四方，眼中射出两道金光，直冲霄汉，惊动了天庭之上的玉皇大帝。这便是日后大闹三界的齐天大圣孙悟空的降世。", "sound_effect": "巨石迸裂声，金光破空声", "characters": ["孙悟空"]},
            {"visual_prompt": "Jade Emperor on throne in heavenly palace, golden clouds, officials around, surprised expressions, Chinese celestial court painting", "audio_script": "玉帝驾坐金阙云宫灵霄宝殿，忽见下界金光冲天，即命千里眼顺风耳查看。二神回报是花果山石猴降生，金光将息。玉帝垂赐恩慈，说道下方之物乃天地精华所生，不足为奇。", "sound_effect": "仙乐缥缈，朝堂钟磬声", "characters": ["玉帝"]},
            {"visual_prompt": "Stone monkey playing with a group of macaques on a mountain, waterfall in background, lush green forest, lively scene", "audio_script": "那石猴在山中与群猴玩耍，日日腾云驾雾，林中采花觅果。一日众猴避暑于松阴之下，戏耍之间顺着涧水寻至源头，只见一帘瀑布飞泻而下，如白练悬空，蔚为壮观。", "sound_effect": "猴群嬉闹声，瀑布水流声", "characters": ["孙悟空", "群猴"]},
            {"visual_prompt": "A brave monkey leaping through a waterfall into a hidden cave, water droplets sparkling, dynamic action, Chinese painting", "audio_script": "一猴高喊：哪个有本事的钻进去寻个源头出来，不伤身体者，我等拜他为王。那石猴应声高叫我进去，纵身一跃，跳入瀑布之中。睁眼一看，里面竟是一座天造地设的洞天福地。", "sound_effect": "跃水声，水珠飞溅", "characters": ["孙悟空", "群猴"]},
            {"visual_prompt": "Inside Water Curtain Cave, stone bridge, stone pots and bowls, lush interior, monkeys celebrating, warm light", "audio_script": "石猴见洞内石座石床一应俱全，中间一块石碣上刻着花果山福地水帘洞洞天。他喜不自胜，复出而呼众猴入内安家。众猴拜他为王，自此石猴登位为美猴王，享乐数百年。", "sound_effect": "群猴欢呼，洞内回声", "characters": ["孙悟空", "群猴"]},
            {"visual_prompt": "Monkey King sitting alone on a rock looking melancholic at sunset, contemplative mood, autumn leaves falling", "audio_script": "美猴王享乐天真何期有三五百载。一日与众猴宴饮，忽然堕下泪来。众猴惊问其故，猴王叹道我虽欢乐，却忧生死之事，将来年老血衰，一旦身死，岂不枉生世界之中。", "sound_effect": "秋风萧瑟，猿啼声", "characters": ["孙悟空", "群猴"]},
            {"visual_prompt": "Two monkey elders advising the Monkey King, pointing toward the sea, a small raft on the shore, dawn light", "audio_script": "众猴中有两个通臂老猴上前说道，大王若要长生，唯有去寻佛仙神圣，学个长生不老之术。猴王闻言大喜，当即伐木扎筏，准备渡海远行，寻访仙道，以求超脱轮回。", "sound_effect": "海浪拍岸，伐木声", "characters": ["孙悟空", "通臂老猴"]},
            {"visual_prompt": "Monkey King on a small raft sailing across vast ocean, waves, dramatic sky, lonely journey, epic voyage", "audio_script": "美猴王独自登筏，乘着东南风，径向西北而行。海中风大浪急，他却不惧，一心只想着寻仙访道。飘摇数日，终于到达南赡部洲地界，弃筏登岸，只见人烟稠密，市井繁华。", "sound_effect": "海浪轰鸣，风声", "characters": ["孙悟空"]},
            {"visual_prompt": "Monkey King disguised in human clothes walking through an ancient Chinese market, people staring, humorous scene", "audio_script": "猴王入得市中，学人穿衣戴帽，学人说话行走。市人见他形貌古怪，纷纷围观。他买些盐酱果饼，也学着人样吃喝。这般在市井间游荡数年，却始终寻不到真仙，心中不免焦急。", "sound_effect": "市井喧嚣，叫卖声", "characters": ["孙悟空"]},
            {"visual_prompt": "A woodcutter singing while chopping wood in a mountain forest, Monkey King listening nearby, serene mountain scene", "audio_script": "一日行至一座大山，忽听林中有人唱歌，词意清雅含仙机。猴王循声寻去，见一樵夫在砍柴。樵夫说此山名灵台方寸山，山中斜月三星洞有一位菩提祖师，能教长生之术。", "sound_effect": "樵歌悠扬，斧斤声", "characters": ["孙悟空", "樵夫"]},
            {"visual_prompt": "Lingtai Fangcun Mountain, Slanted Moon Three Star Cave entrance, mist and clouds, immortal atmosphere, grand gate", "audio_script": "猴王大喜，依樵夫指引寻至洞府门前。只见那山势嵯峨，洞门高耸，上书灵台方寸山斜月三星洞。正欲叩门，门呀然洞开，一个仙童走出，说是祖师唤他进去。", "sound_effect": "仙乐隐隐，门轴声", "characters": ["孙悟空", "仙童"]},
            {"visual_prompt": "Bodhi Patriarch on a dais teaching, disciples around, Monkey King kneeling before him, solemn Taoist hall", "audio_script": "猴王入内拜见菩提祖师。祖师问他姓名乡贯，猴王答道我无性，人若骂我我也不恼。祖师闻言大喜，说这般却好。又问其来路，猴王备述花果山石猴来历，祖师暗暗称奇。", "sound_effect": "道堂静穆，木鱼声", "characters": ["孙悟空", "菩提祖师"]},
            {"visual_prompt": "Bodhi Patriarch naming the monkey, writing characters, golden light around the name Wukong, ceremonial moment", "audio_script": "祖师道我门中有十二字分派起名，到你乃第十辈之悟字。与你起个法名叫做孙悟空，好么？猴王欢喜道好今日方知有姓有名。自此众猴称他为孙悟空，又叫他孙长老。", "sound_effect": "钟磬齐鸣，诵经声", "characters": ["孙悟空", "菩提祖师"]},
            {"visual_prompt": "Sun Wukong studying scriptures and practicing meditation in a Taoist temple, diligent learning scene, candlelight", "audio_script": "悟空自此在洞中修行，与众师兄讲经论道，习字焚香，洒扫应对。如此在洞府中过了六七年，一日祖师升坛讲道，悟空在旁听讲，喜得抓耳挠腮，眉开眼笑，显出悟性非凡。", "sound_effect": "讲经声，翻经卷声", "characters": ["孙悟空", "菩提祖师"]},
            {"visual_prompt": "Bodhi Patriarch tapping Sun Wukong on the head three times at night, secret transmission scene, moonlight through window", "audio_script": "祖师见悟空悟性超群，欲秘传大道。一日登坛，在悟空头上敲了三下，倒背着手走入里面。悟空心领神会，当夜三更时分，从后门潜入祖师寝榻之前，跪求长生妙诀。", "sound_effect": "夜静更深，更鼓声", "characters": ["孙悟空", "菩提祖师"]},
            {"visual_prompt": "Sun Wukong practicing magical transformations, turning into various forms, mystical energy swirling, dynamic poses", "audio_script": "祖师大喜，遂传与悟空长生大法，又教他七十二般变化与筋斗云之术。那筋斗云一跃十万八千里。悟空日夜苦练，渐渐神通广大，腾云驾雾，变化无穷，远胜同门师兄弟。", "sound_effect": "风雷之声，变化灵音", "characters": ["孙悟空"]},
            {"visual_prompt": "Sun Wukong showing off transformations to fellow disciples, turning into a pine tree, crowd laughing, festive scene", "audio_script": "一日众师兄要悟空变化一试。悟空卖弄神通，变作一棵松树，众人鼓掌大笑。却不料惊动了祖师。祖师唤悟空上前，面色一沉，说道你这般卖弄，久后必惹祸端，速速回去罢。", "sound_effect": "众人哄笑，变法灵音", "characters": ["孙悟空", "菩提祖师"]},
            {"visual_prompt": "Sun Wukong kneeling farewell to Bodhi Patriarch at cave entrance, emotional parting, sunset, ominous clouds gathering", "audio_script": "祖师道你此去定生不良，凭你怎么惹祸，却不许说是我的徒弟。悟空含泪拜别，踏上筋斗云，须臾间便回到了花果山。然而祖师那意味深长的警告犹在耳畔，一场惊天动地的大祸正在酝酿。", "sound_effect": "离愁别绪，隐隐雷声", "characters": ["孙悟空", "菩提祖师"]},
        ]
        ep2 = [
            {"visual_prompt": "Sun Wukong descending on a somersault cloud to Flower Fruit Mountain, monkeys cheering below, triumphant return", "audio_script": "悟空按下云头，落在花果山水帘洞前。众猴见大王归来，纷纷跪拜哭泣，诉说自从大王去后，有个混世魔王来占洞府，抢去许多器物。悟空闻言大怒，誓要扫除妖魔，重整山门。", "sound_effect": "云霞破空，猴群哭诉", "characters": ["孙悟空", "群猴"]},
            {"visual_prompt": "Sun Wukong confronting the Demon King in a dark cave, fierce standoff, weapons raised, dramatic tension", "audio_script": "悟空问明魔王住处，纵起筋斗云直抵水脏洞前。那混世魔王手持大刀，凶神恶煞而出，见是个矮小猴子，心中轻视。二人言语不和，便动起手来，刀光棒影，杀得天昏地暗。", "sound_effect": "兵器交击，呼喝声", "characters": ["孙悟空", "混世魔王"]},
            {"visual_prompt": "Sun Wukong defeating the Demon King, plucking hairs that turn into tiny monkeys overwhelming the enemy, magical battle", "audio_script": "悟空拔下一把毫毛，吹口仙气，变作数百个小猴，蜂拥而上，将魔王团团围住。那魔王抵挡不住，被悟空一棒打倒，化作一阵清风散去。悟空救出被掳的猴众，得胜而归。", "sound_effect": "猴群呐喊，仙风声", "characters": ["孙悟空", "混世魔王"]},
            {"visual_prompt": "Sun Wukong training monkeys with wooden weapons on a mountain clearing, military drill scene, flags and formations", "audio_script": "悟空既归，恐众猴武艺不精，难保山门。遂教群猴砍竹为枪，削木为刀，操演阵法。然而操练日久，悟空忧道竹木刀枪不坚，遇真敌难以取胜，必须寻些钢铁打造称手兵器。", "sound_effect": "操练号令，兵器碰撞", "characters": ["孙悟空", "群猴"]},
            {"visual_prompt": "Four old monkeys advising Sun Wukong, pointing toward the eastern sea, Dragon Palace in the distance, council scene", "audio_script": "悟空正忧虑间，四个老猴上前献策。说道大王若要兵器，水帘洞桥东之龙宫，乃是东海敖广所居，兵器极多，可往求取。悟空大喜，当即分水直入东海，径奔龙宫而去。", "sound_effect": "议事声，海涛隐约", "characters": ["孙悟空", "老猴"]},
            {"visual_prompt": "Sun Wukong parting the sea waters, walking on the ocean floor toward a glowing Dragon Palace, majestic underwater scene", "audio_script": "悟空使个避水法，分开海水，大步而行。只见海底珊瑚璀璨，珠光宝气，一座水晶宫阙巍然矗立。巡海夜叉见了急忙通报，东海龙王敖广率众出迎，请入宫中款待。", "sound_effect": "分水之声，海底仙乐", "characters": ["孙悟空", "敖广"]},
            {"visual_prompt": "Dragon King presenting a giant halberd to Sun Wukong in the treasure hall, golden weapons displayed, negotiation scene", "audio_script": "龙王命取出一杆三千六百斤的九股叉递与悟空。悟空接过舞了一回，嫌太轻了。龙王又换七千二百斤的方天戟，悟空仍嫌轻。龙王面露难色，宫中已无更重的兵器可献。", "sound_effect": "兵器舞动声，龙宫钟磬", "characters": ["孙悟空", "敖广"]},
            {"visual_prompt": "Dragon Queen and princess whispering to Dragon King about a glowing iron pillar in the sea, plotting scene, opulent hall", "audio_script": "龙婆龙女上前密语道大王宫中那块天河定底的神珍铁，这几日霞光万道，莫非是这猴子的缘分？龙王犹豫道那是大禹治水留下的定海神针，重达一万三千五百斤，怕他拿不动。", "sound_effect": "密语声，珠玉碰撞", "characters": ["敖广", "龙婆"]},
            {"visual_prompt": "Sun Wukong finding the glowing Ruyi Jingu Bang pillar in the deep sea, golden radiance, treasure discovery, epic", "audio_script": "悟空随龙王来到海底，见一根金光灿灿的铁柱。他上前一把攥住说再细些短些才好。话音未落，那宝物竟应声而短，依他所愿。悟空大喜，原来是如意金箍棒，可大可小，随心变化。", "sound_effect": "宝物鸣响，金光破散", "characters": ["孙悟空"]},
            {"visual_prompt": "Sun Wukong triumphantly wielding the golden staff, showing it off to dragons, demanding more treasure, imposing", "audio_script": "悟空得了金箍棒，舞动如风，喜不自胜。却又对龙王道好是好，只是我赤手空拳来，如今有了兵器，还少一副披挂。劳烦龙王再送我一副，否则我便不动这金箍棒了。", "sound_effect": "金箍棒风声，龙宫震动", "characters": ["孙悟空", "敖广"]},
            {"visual_prompt": "Three Dragon Kings arriving with golden armor, presenting it to Sun Wukong, assembly of dragons, ceremonial", "audio_script": "敖广无奈，急忙撞钟击鼓，唤来南海敖钦北海敖顺西海敖闰三位兄弟。四位龙王齐聚，凑了一副藕丝步云履锁子黄金甲凤翅紫金冠，齐齐奉上，悟空穿戴齐整，金光护体。", "sound_effect": "钟鼓齐鸣，龙吟声", "characters": ["孙悟空", "敖广"]},
            {"visual_prompt": "Sun Wukong flying out of the sea with golden armor and staff, dragons watching helplessly, departure scene", "audio_script": "悟空披挂整齐，手持金箍棒，得意洋洋。他纵起筋斗云，跳出东海，径回花果山。四位龙王面面相觑，又怒又惧，商议道这猴精如此蛮横，夺我镇海之宝，定要上天庭告他一状。", "sound_effect": "云霞破空，龙宫哀叹", "characters": ["孙悟空", "敖广"]},
            {"visual_prompt": "Sun Wukong back on Flower Fruit Mountain showing the staff to monkeys, celebratory feast, joyous scene", "audio_script": "悟空回到水帘洞，向众猴展示金箍棒与披挂。将棒变作绣花针藏于耳内，又变大如擎天柱，腾挪变化，众猴看得目眩神迷。当即大摆筵席，与群猴痛饮，庆贺得了神兵。", "sound_effect": "欢呼宴饮，金棒舞动", "characters": ["孙悟空", "群猴"]},
            {"visual_prompt": "Sun Wukong feasting and drinking wine, then falling asleep, a spirit chain emerging from his body, dreamlike", "audio_script": "酒至半酣，悟空沉沉睡去。忽然魂魄离体，被两个勾魂使者拿住，竟锁到了幽冥界。悟空大怒道我老孙已修长生，跳出三界，你这勾魂的怎敢拿我！一脚踢翻使者，闯入森罗殿。", "sound_effect": "阴风阵阵，铁链声", "characters": ["孙悟空"]},
            {"visual_prompt": "Sun Wukong storming the Underworld palace of King Yan, confronting judges, ledgers of life and death, dark realm", "audio_script": "悟空闯入森罗殿，十代阎君大惊。悟空喝道老孙超出三界之外，不在五行之中，为何勾我？阎王支吾道可能是同名同姓之误。悟空取过生死簿，索要笔来，将自己名字一笔勾销。", "sound_effect": "怒喝声，翻簿声", "characters": ["孙悟空", "阎王"]},
            {"visual_prompt": "Sun Wukong crossing out names in the Book of Life and Death, monkeys souls freed, dramatic underworld scene", "audio_script": "悟空翻开生死簿，见猴属之类众多，索性将猴属之名一概勾销。自此花果山众猴皆不受阎王管辖。十代阎王面如土色，齐齐告饶。悟空方才丢下簿子，打出幽冥界，魂归本体。", "sound_effect": "笔走如飞，阴界惊呼", "characters": ["孙悟空", "阎王"]},
            {"visual_prompt": "Dragon Kings and Underworld Kings meeting in heaven, presenting memorials to the Jade Emperor, formal court scene", "audio_script": "东海龙王与地藏王菩萨各自修表，上达天庭，告那妖猴夺宝销名之罪。玉帝览奏，问众仙卿这妖猴是何来历。太白金星出班奏道这猴乃天地精华所生，不如降一道招安圣旨，免动刀兵。", "sound_effect": "朝堂钟磬，奏表展开", "characters": ["玉帝", "太白金星"]},
            {"visual_prompt": "Jade Emperor listening to ministers in the Lingxiao Palace, debating how to handle the monkey king, grand court", "audio_script": "玉帝沉吟道依卿所奏。即命文曲星官修诏，着太白金星赍捧招安。太白金星领旨，告别众仙，驾云直下凡间，往花果山水帘洞去宣悟空上天受禄，一场天庭与花果山的风波就此展开。", "sound_effect": "圣旨宣读，仙乐送行", "characters": ["玉帝", "太白金星"]},
            {"visual_prompt": "Sun Wukong atop a mountain looking up at descending clouds, the Jade Emperor's decree arriving, omen of destiny, cliffhanger", "audio_script": "悟空正在山中操练，忽见天空祥云缭绕，一位白发仙官飘然而至。那正是太白金星奉旨前来。悟空仰望天际，心中既惊又喜。然而他尚不知，这一道招安圣旨，将引出日后大闹天宫的滔天大祸。", "sound_effect": "仙乐渐近，悬念雷声", "characters": ["孙悟空", "太白金星"]},
        ]
        return [ep1, ep2]

    @staticmethod
    def _generate_http(handle, messages: list[dict], temperature: float) -> str:
        # 8192 经验值：Qwen3.5-9B 对复杂编剧 prompt 会先输出约 7-8k token 的
        # "Thinking Process:" 再吐 JSON；16384 会让模型过度思考触顶无 JSON，8192 稳定产出。
        max_tokens = int(os.environ.get("PROMPT_EXPAND_MAX_TOKENS", "8192"))
        content, _ = handle.client.chat(
            messages,
            model=handle.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        return content

    @staticmethod
    def _strip_thinking(text: str) -> str:
        end_tag = chr(60) + "/" + "think" + chr(62)
        if end_tag in text:
            text = text.split(end_tag, 1)[1]
        tp_match = re.search(r"thinking process:", text, re.IGNORECASE)
        if tp_match:
            after = text[tp_match.end():]
            json_match = re.search(r'\{[\s\n]*"scenes"\s*:', after)
            if json_match:
                text = after[json_match.start():]
            else:
                brace = after.find("{")
                if brace >= 0:
                    text = after[brace:]
                else:
                    logger.warning("thinking truncated, no JSON after thinking process:, fallback to full-text scenes search")
                    full_match = re.search(r'\{[\s\n]*"scenes"\s*:', text)
                    if full_match:
                        text = text[full_match.start():]
                    else:
                        text = after
        return text.strip()

    @staticmethod
    def _parse_json(content: str) -> dict | list:
        text = PromptExpandStage._strip_thinking(content.strip())
        text = re.sub(r'"(\w+):\s*"', r'"\1": "', text)
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                inner = parts[1]
                if inner.lower().startswith("json"):
                    inner = inner[4:]
                text = inner.strip()
            else:
                text = text.strip("`")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\n]*"scenes"\s*:', text)
            if json_match:
                candidate = text[json_match.start():]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    for end in range(len(candidate) - 1, max(len(candidate) - 2000, 0), -1):
                        if candidate[end] == '}':
                            try:
                                return json.loads(candidate[:end + 1])
                            except json.JSONDecodeError:
                                continue
            brace = text.find("{")
            if brace >= 0:
                candidate = text[brace:]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    for end in range(len(candidate) - 1, max(len(candidate) - 2000, 0), -1):
                        if candidate[end] == '}':
                            try:
                                return json.loads(candidate[:end + 1])
                            except json.JSONDecodeError:
                                continue
            raw_match = re.search(r'\{[\s\n]*"scenes"\s*:', content)
            if raw_match:
                candidate = content[raw_match.start():]
                for end in range(len(candidate) - 1, max(len(candidate) - 8000, 0), -1):
                    if candidate[end] == '}':
                        try:
                            return json.loads(candidate[:end + 1])
                        except json.JSONDecodeError:
                            continue
            raise
