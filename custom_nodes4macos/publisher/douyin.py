from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("custom_nodes4macos.publisher.douyin")

_DOUYIN_UPLOAD_URL = "https://creator.douyin.com/creator-micro/content/upload"


class PublishConfigError(RuntimeError):
    pass


class PublishDependencyError(RuntimeError):
    pass


@dataclass
class PublishMeta:
    title: str
    tags: list[str] = field(default_factory=list)
    cover_path: str = ""


@dataclass
class PublishResult:
    status: str
    platform: str = "douyin"
    video_path: str = ""
    title: str = ""
    tags: list[str] = field(default_factory=list)
    draft_url: str = ""
    manifest_path: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "platform": self.platform,
            "video_path": self.video_path,
            "title": self.title,
            "tags": self.tags,
            "draft_url": self.draft_url,
            "manifest_path": self.manifest_path,
            "error": self.error,
        }


# fork-and-strip 自 social-auto-upload，仅保留 Douyin 草稿上传。
# dry_run=True（默认）：只写清单不触网，安全可逆。
# dry_run=False：需 cookies_path + playwright（缺一抛对应异常，由调用方降级）。
class DouyinPublisher:

    def upload_draft(
        self,
        video_path: str,
        meta: PublishMeta,
        cookies_path: str,
        dry_run: bool = True,
        manifest_dir: str = "",
    ) -> PublishResult:
        if not video_path or not os.path.isfile(video_path):
            raise PublishConfigError(f"video not found: {video_path}")

        manifest_path = self._manifest_path(manifest_dir, video_path)

        if dry_run:
            logger.info(
                "douyin dry_run: video=%s title=%s tags=%s",
                video_path, meta.title, meta.tags,
            )
            result = PublishResult(
                status="dry_run",
                video_path=video_path,
                title=meta.title,
                tags=list(meta.tags),
                manifest_path=manifest_path,
            )
            self._write_manifest(manifest_path, result)
            return result

        if not cookies_path or not os.path.isfile(cookies_path):
            raise PublishConfigError(f"cookies not found: {cookies_path}")

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise PublishDependencyError(
                "playwright not installed; run: pip install playwright && playwright install chromium"
            ) from exc

        return self._upload_live(sync_playwright, video_path, meta, cookies_path, manifest_path)

    def _upload_live(
        self,
        sync_playwright,
        video_path: str,
        meta: PublishMeta,
        cookies_path: str,
        manifest_path: str,
    ) -> PublishResult:
        logger.info("douyin live upload: video=%s title=%s", video_path, meta.title)
        cookies = self._load_cookies(cookies_path)
        draft_url = ""

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            try:
                context = browser.new_context()
                context.add_cookies(cookies)
                page = context.new_page()
                page.goto(_DOUYIN_UPLOAD_URL, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(3000)
                logger.info("douyin: upload page loaded")

                file_input = page.locator('input[type="file"]').first
                file_input.set_input_files(video_path)
                logger.info("douyin: video file set, waiting for upload...")
                page.wait_for_timeout(15000)

                title_box = page.locator('[contenteditable="true"]').first
                if title_box.count() > 0:
                    title_box.fill("")
                    title_box.type(meta.title, delay=50)
                    logger.info("douyin: title filled")

                for tag in meta.tags:
                    try:
                        tag_input = page.get_by_placeholder("话题").first
                        tag_input.type(tag, delay=50)
                        page.wait_for_timeout(800)
                        page.keyboard.press("Enter")
                    except Exception:
                        logger.warning("douyin: tag '%s' input skipped", tag)

                draft_url = self._click_save_draft(page)
            finally:
                browser.close()

        result = PublishResult(
            status="draft_saved" if draft_url else "uploaded_unconfirmed",
            video_path=video_path,
            title=meta.title,
            tags=list(meta.tags),
            draft_url=draft_url,
            manifest_path=manifest_path,
        )
        self._write_manifest(manifest_path, result)
        logger.info("douyin: live upload done status=%s draft_url=%s", result.status, draft_url)
        return result

    @staticmethod
    def _click_save_draft(page) -> str:
        for text in ("存草稿", "保存草稿"):
            btn = page.get_by_text(text, exact=False).first
            if btn.count() > 0:
                btn.click()
                page.wait_for_timeout(3000)
                logger.info("douyin: draft saved via '%s'", text)
                return page.url
        btn = page.locator("button:has-text('草稿')").first
        if btn.count() > 0:
            btn.click()
            page.wait_for_timeout(3000)
            logger.info("douyin: draft saved via button:has-text('草稿')")
            return page.url
        logger.warning("douyin: save-draft button not found, leaving upload unsaved")
        return ""

    @staticmethod
    def _load_cookies(cookies_path: str) -> list[dict]:
        with open(cookies_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cookies: list[dict] = []
        for item in raw:
            c = {
                "name": item.get("name", ""),
                "value": item.get("value", ""),
                "domain": item.get("domain", ".douyin.com"),
                "path": item.get("path", "/"),
            }
            if item.get("expires"):
                c["expires"] = item["expires"]
            cookies.append(c)
        logger.info("douyin: loaded %d cookies from %s", len(cookies), cookies_path)
        return cookies

    @staticmethod
    def _manifest_path(manifest_dir: str, video_path: str) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0] or "video"
        d = manifest_dir or os.path.dirname(video_path) or "."
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{base}_douyin_publish.json")

    @staticmethod
    def _write_manifest(manifest_path: str, result: PublishResult) -> None:
        payload = result.to_dict()
        payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("douyin manifest written: %s", manifest_path)
