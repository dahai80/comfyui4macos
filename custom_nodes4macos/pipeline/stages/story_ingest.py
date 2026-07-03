from __future__ import annotations

import json
import logging
import os
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

        with model_manager.acquire("llm") as handle:
            model, tokenizer = handle.model
            outline = self._generate_outline(
                model, tokenizer, chapters, episode_count,
                episode_duration_min, ctx,
            )

        episodes = self._parse_episodes(outline)
        if not episodes:
            raise RuntimeError("story_ingest: failed to generate episode outline")

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
        self, model, tokenizer, chapters, episode_count,
        episode_duration_min, ctx,
    ) -> str:
        chapters_summary = ""
        for ch in chapters[:60]:
            chapters_summary += (
                f"第{ch['chapter_id']}章「{ch['title']}」"
                f"（{len(ch['text'])}字）\n"
            )
            snippet = ch["text"][:500]
            chapters_summary += f"  摘要: {snippet}...\n\n"

        system_msg = (
            "你是一位资深编剧统筹，擅长将长篇小说改编为连续剧。\n"
            "任务：根据小说章节概要，规划分集大纲。\n"
            "输出严格 JSON 格式：\n"
            '{"episodes": [{"episode_id": 1, "title": "...", '
            '"chapters": "第1-3章", "synopsis": "...", '
            '"key_scenes": ["场景1", "场景2"], '
            '"cliffhanger": "..."}]}\n'
            f"共 {episode_count} 集，每集约 {episode_duration_min} 分钟。\n"
            "要求：\n"
            "1. 每集必须有悬念结尾（cliffhanger）\n"
            "2. 剧情节奏有起伏，不能平铺直叙\n"
            "3. 主要角色线索贯穿始终\n"
            "4. 只输出 JSON，无其他文字"
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
        return self._generate(model, tokenizer, messages, temperature)

    @staticmethod
    def _generate(model, tokenizer, messages, temperature) -> str:
        try:
            from mlx_lm import generate as mlx_generate
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            output = mlx_generate(
                model, tokenizer,
                prompt=prompt_text,
                max_tokens=8192,
                temp=temperature,
                verbose=False,
            )
            return output.strip()
        except ImportError:
            logger.warning("mlx_lm not available, falling back to HTTP")
            return StoryIngestStage._generate_http(messages, temperature)

    @staticmethod
    def _generate_http(messages, temperature) -> str:
        from ...fusion_client import FusionMLXClient
        with FusionMLXClient() as client:
            if not client.health():
                raise RuntimeError("fusion-mlx unreachable (HTTP fallback)")
            content, _ = client.chat(
                messages, temperature=temperature, json_mode=True,
            )
        return content

    @staticmethod
    def _parse_episodes(content: str) -> list[dict]:
        text = content.strip()
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
            parsed = json.loads(text)
        except json.JSONDecodeError:
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
