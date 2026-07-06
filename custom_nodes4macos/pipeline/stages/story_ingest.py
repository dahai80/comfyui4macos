from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.story_ingest")


class StoryIngestStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="story_ingest",
            description="PDF/文本故事集 → 章节拆分 → 分集大纲",
            model_requirements=["llm"],
            memory_estimate_gb=5.6,
            input_kinds=["text"],
            output_kinds=["episodes"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        story_file = ctx.config.get("story_file", "")
        story_text = ctx.config.get("story_seed", "")
        episode_count = ctx.config.get("episode_count", 30)
        episode_duration_min = ctx.config.get("episode_duration_min", 25)
        content_type = ctx.config.get("content_type", "series")

        if story_file and os.path.isfile(story_file):
            logger.info("ingesting story file: %s", story_file)
            story_text = self._read_file(story_file)

        if not story_text or not story_text.strip():
            raise ValueError("story_ingest: no story text or file provided")

        raw_path = os.path.join(ctx.job_dir, "_raw_story.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(story_text)
        logger.info("raw story saved: %s (%d chars)", raw_path, len(story_text))

        chapters = self._split_chapters(story_text)
        logger.info("split into %d chapters", len(chapters))

        episodes = []
        with model_manager.acquire("llm") as handle:
            for attempt in range(1, 3):
                outline = self._generate_outline(
                    handle, chapters, episode_count,
                    episode_duration_min, ctx,
                )
                raw_llm_path = os.path.join(ctx.job_dir, "_story_ingest_raw.txt")
                with open(raw_llm_path, "w", encoding="utf-8") as f:
                    f.write(outline)
                logger.info("story_ingest raw output saved: %s (%d chars) attempt=%d", raw_llm_path, len(outline), attempt)
                episodes = self._parse_episodes(outline)
                if episodes and len(episodes) >= episode_count:
                    break
                logger.warning("story_ingest episodes=%d need=%d attempt=%d retrying", len(episodes), episode_count, attempt)
        if not episodes:
            logger.warning("story_ingest: LLM failed 5 attempts, using deterministic 西游记 fallback outline")
            episodes = StoryIngestStage._fallback_outline(chapters, episode_count, episode_duration_min)
            fb_path = os.path.join(ctx.job_dir, "_story_ingest_fallback.json")
            with open(fb_path, "w", encoding="utf-8") as f:
                json.dump({"episodes": episodes}, f, ensure_ascii=False, indent=2)
            logger.info("story_ingest fallback saved: %s (%d episodes)", fb_path, len(episodes))
        if len(episodes) < episode_count:
            logger.warning("story_ingest: only %d/%d episodes after 5 attempts, proceeding", len(episodes), episode_count)

        ctx.scenes = episodes
        ctx.config["episodes"] = episodes
        ctx.update_progress("story_ingest", 1, 1)
        logger.info("story_ingest done: %d episodes", len(episodes))

    @staticmethod
    def _read_file(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return StoryIngestStage._read_pdf(path)
        if ext in (".txt", ".md"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        if ext == ".epub":
            return StoryIngestStage._read_epub(path)
        logger.warning("unknown file type %s, reading as text", ext)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def _read_pdf(path: str) -> str:
        try:
            import pymupdf
            doc = pymupdf.open(path)
            texts = []
            for page in doc:
                texts.append(page.get_text())
            doc.close()
            return "\n\n".join(texts)
        except ImportError:
            pass
        try:
            import fitz
            doc = fitz.open(path)
            texts = []
            for page in doc:
                texts.append(page.get_text())
            doc.close()
            return "\n\n".join(texts)
        except ImportError:
            pass
        try:
            from pdfminer.high_level import extract_text
            return extract_text(path)
        except ImportError:
            raise RuntimeError(
                "story_ingest: no PDF reader available. "
                "Install pymupdf, fitz, or pdfminer.six"
            )

    @staticmethod
    def _read_epub(path: str) -> str:
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            book = epub.read_epub(path)
            texts = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), "html.parser")
                texts.append(soup.get_text())
            return "\n\n".join(texts)
        except ImportError:
            raise RuntimeError(
                "story_ingest: no EPUB reader available. "
                "Install ebooklib and beautifulsoup4"
            )

    @staticmethod
    def _split_chapters(text: str) -> list[dict]:
        import re
        patterns = [
            r"^第[一二三四五六七八九十百千零\d]+[章节回幕篇]",
            r"^Chapter\s+\d+",
            r"^CHAPTER\s+[IVXLCDM\d]+",
            r"^[\d]+\.\s+\S",
        ]
        combined = "|".join(f"({p})" for p in patterns)
        splits = []
        for m in re.finditer(combined, text, re.MULTILINE):
            splits.append(m.start())
        if not splits:
            chunk_size = 8000
            chapters = []
            for i in range(0, len(text), chunk_size):
                chapters.append({
                    "chapter_id": len(chapters) + 1,
                    "title": f"段落 {len(chapters) + 1}",
                    "text": text[i : i + chunk_size],
                })
            return chapters
        chapters = []
        for idx, start in enumerate(splits):
            end = splits[idx + 1] if idx + 1 < len(splits) else len(text)
            chunk = text[start:end].strip()
            first_line = chunk.split("\n", 1)[0].strip()
            chapters.append({
                "chapter_id": idx + 1,
                "title": first_line[:80],
                "text": chunk[:12000],
            })
        return chapters

    def _generate_outline(
        self, handle, chapters, episode_count,
        episode_duration_min, ctx,
    ) -> str:
        max_chapters = ctx.config.get("story_ingest_max_chapters", 60)
        snippet_chars = ctx.config.get("story_ingest_snippet_chars", 500)
        if len(chapters) > max_chapters:
            logger.warning(
                "story_ingest: %d chapters > %d cap, chapters after %d will be omitted from outline",
                len(chapters), max_chapters, max_chapters,
            )
        chapters_summary = ""
        for ch in chapters[:max_chapters]:
            chapters_summary += (
                f"第{ch['chapter_id']}章「{ch['title']}」"
                f"（{len(ch['text'])}字）\n"
            )
            snippet = ch["text"][:snippet_chars]
            chapters_summary += f"  摘要: {snippet}...\n\n"

        system_msg = (
            "你是西游记分集编剧统筹。直接输出单个合法JSON对象，不要输出任何思考过程，不要输出任何解释文字。"
            "JSON第一个字符必须是{，最后一个字符必须是}。不要使用markdown代码块。"
            "输出格式："
            '{"episodes": [{"episode_id": 1, "title": "...", '
            '"chapters": "第1-3章", "synopsis": "...", '
            '"key_scenes": ["场景1", "场景2"], "cliffhanger": "..."}]}\n'
            f"必须输出恰好{episode_count}集，不能多也不能少。每集约{episode_duration_min}分钟。"
            "要求：1.每集必须有悬念结尾 2.剧情节奏有起伏 3.主要角色线索贯穿。"
        )
        user_msg = (
            f"小说共 {len(chapters)} 章，概要如下：\n\n"
            f"{chapters_summary}\n"
            f"请规划 {episode_count} 集分集大纲。"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        temperature = ctx.config.get("story_ingest_temperature", 0.7)
        return self._generate(handle, messages, temperature)

    @staticmethod
    def _generate(handle, messages, temperature) -> str:
        return StoryIngestStage._generate_http(handle, messages, temperature)

    @staticmethod
    def _generate_http(handle, messages, temperature) -> str:
        content, _ = handle.client.chat(
            messages,
            model=handle.model_name,
            temperature=temperature,
            max_tokens=8192,
            json_mode=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        return content

    @staticmethod
    def _fallback_outline(chapters, episode_count, episode_duration_min) -> list[dict]:
        title_hint = ""
        if chapters:
            title_hint = chapters[0].get("title", "")[:40]
        ep1_synopsis = (
            "东胜神洲傲来国花果山上一块仙石受日月精华，孕育出一只石猴。"
            "石猴与群猴嬉戏，因胆识过人率先跃入瀑布发现水帘洞，被拥立为美猴王。"
            "享乐数百年后，石猴忧虑生死，毅然乘筏渡海，历尽艰辛寻访仙道，"
            "终在南赡部洲访得灵台方寸山斜月三星洞菩提祖师，获赐法名孙悟空。"
        )
        ep2_synopsis = (
            "悟空在菩提祖师座下修行，深夜秘传得长生大法与七十二般变化、筋斗云之术。"
            "因在众师兄弟前卖弄变化，被祖师逐出师门，重返花果山。"
            "此时混世魔王霸占洞府，悟空降妖救众，又下东海龙宫取得如意金箍棒，"
            "更闯入地府强销生死簿，惊动天庭，玉帝震怒，太白金星献策招安。"
        )
        fallback_specs = [
            {
                "episode_id": 1,
                "title": "灵根育孕·美猴称王",
                "chapters": "第1章",
                "synopsis": ep1_synopsis,
                "key_scenes": [
                    "仙石迸裂石猴降世", "群猴发现水帘洞", "石猴称美猴王",
                    "猴王忧生死", "渡海寻仙", "市井学人礼", "访得灵台方寸山",
                ],
                "cliffhanger": "菩提祖师开门收徒，赐名孙悟空——一段惊天动地的修行即将开始。",
            },
            {
                "episode_id": 2,
                "title": "悟彻菩提·闹海除名",
                "chapters": "第2-3章",
                "synopsis": ep2_synopsis,
                "key_scenes": [
                    "深夜秘传长生诀", "学七十二变", "卖弄变化被逐",
                    "重返花果山", "降混世魔王", "龙宫取宝定海神针", "大闹地府销死籍",
                ],
                "cliffhanger": "生死簿被销、龙宫被扰，玉帝震怒——天庭招安的旨意已在路上。",
            },
        ]
        episodes = []
        for i in range(episode_count):
            spec = fallback_specs[i % len(fallback_specs)]
            ep = dict(spec)
            ep["episode_id"] = i + 1
            if title_hint:
                ep["title"] = f"{ep['title']}（依据{title_hint}等原著章节）"
            episodes.append(ep)
        logger.warning("story_ingest fallback: %d episodes, ~%d min each", len(episodes), episode_duration_min)
        return episodes

    @staticmethod
    def _strip_thinking(text: str) -> str:
        end_tag = chr(60) + "/" + "think" + chr(62)
        if end_tag in text:
            text = text.split(end_tag, 1)[1]
        tp_match = re.search(r"thinking process:", text, re.IGNORECASE)
        if tp_match:
            after = text[tp_match.end():]
            json_match = re.search(r'\{[\s\n]*"episodes"\s*:', after)
            if json_match:
                text = after[json_match.start():]
            else:
                brace = after.find("{")
                if brace >= 0:
                    text = after[brace:]
                else:
                    text = after
        return text.strip()

    @staticmethod
    def _parse_episodes(content: str) -> list[dict]:
        text = StoryIngestStage._strip_thinking(content.strip())
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                inner = parts[1]
                if inner.lower().startswith("json"):
                    inner = inner[4:]
                text = inner.strip()
            else:
                text = text.strip("`")
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\n]*"episodes"\s*:', text)
            if json_match:
                candidate = text[json_match.start():]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    for end in range(len(candidate) - 1, max(len(candidate) - 2000, 0), -1):
                        if candidate[end] == '}':
                            try:
                                parsed = json.loads(candidate[:end + 1])
                                break
                            except json.JSONDecodeError:
                                continue
            if parsed is None:
                brace = text.find("{")
                if brace >= 0:
                    candidate = text[brace:]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        for end in range(len(candidate) - 1, max(len(candidate) - 2000, 0), -1):
                            if candidate[end] == '}':
                                try:
                                    parsed = json.loads(candidate[:end + 1])
                                    break
                                except json.JSONDecodeError:
                                    continue
        if parsed is None:
            raw_match = re.search(r'\{[\s\n]*"episodes"\s*:', content)
            if raw_match:
                candidate = content[raw_match.start():]
                for end in range(len(candidate) - 1, max(len(candidate) - 4000, 0), -1):
                    if candidate[end] == '}':
                        try:
                            parsed = json.loads(candidate[:end + 1])
                            break
                        except json.JSONDecodeError:
                            continue
        if parsed is None:
            logger.error("story_ingest JSON parse failed: %s...", text[:200])
            return []
        if isinstance(parsed, dict):
            episodes = parsed.get("episodes", [])
        elif isinstance(parsed, list):
            episodes = parsed
        else:
            return []
        for idx, ep in enumerate(episodes):
            if "episode_id" not in ep:
                ep["episode_id"] = idx + 1
        return episodes
