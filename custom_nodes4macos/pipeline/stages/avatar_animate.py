from __future__ import annotations

import logging
import math
import os
import wave

import cv2
import numpy as np

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.avatar_animate")


class AvatarAnimateStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="avatar_animate",
            description="数字人动画: 参考帧 + TTS音频 → 面部动画视频（口型+表情）",
            model_requirements=[],
            memory_estimate_gb=2.0,
            input_kinds=["avatar_package", "audio"],
            output_kinds=["clip"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        avatar_dir = ctx.artifacts.get("avatar_package") or ctx.config.get("avatar_package", "")
        ref_path = ctx.artifacts.get("avatar_reference") or ctx.config.get("avatar_reference", "")

        if not ref_path or not os.path.isfile(ref_path):
            ref_path = ctx.config.get("avatar_reference", "")
        if not ref_path or not os.path.isfile(ref_path):
            ref_path = os.path.join(ctx.job_dir, "_avatar", "reference.png")
        if not os.path.isfile(ref_path):
            logger.error("avatar_animate: no reference frame found")
            return

        avatar_meta_path = os.path.join(avatar_dir, "avatar_meta.json") if avatar_dir else ""
        avatar_meta = {}
        if avatar_meta_path and os.path.isfile(avatar_meta_path):
            import json
            with open(avatar_meta_path, "r", encoding="utf-8") as f:
                avatar_meta = json.load(f)

        motion_dir = os.path.join(avatar_dir, "motion_frames") if avatar_dir else ""
        motion_frames = []
        if os.path.isdir(motion_dir):
            for fn in sorted(os.listdir(motion_dir)):
                if fn.endswith(".png") or fn.endswith(".jpg"):
                    motion_frames.append(os.path.join(motion_dir, fn))

        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)

        scenes = ctx.scenes
        if not scenes:
            logger.warning("avatar_animate: no scenes to process")
            return

        for i, scene in enumerate(scenes):
            sid = scene.get("scene_id", i + 1)
            if ctx.has_artifact_on_disk(sid, "clip"):
                logger.info("avatar_animate scene %d: clip exists, skipping", sid)
                continue

            audio_path = ctx.get_artifact(sid, "audio")
            if not audio_path or not os.path.isfile(audio_path):
                logger.warning("avatar_animate scene %d: no audio, skipping", sid)
                continue

            clip_path = ctx.artifact_path(sid, "clip")
            duration = scene.get("duration_seconds", 10)

            self._animate_scene(
                ref_path=ref_path,
                audio_path=audio_path,
                output_path=clip_path,
                duration=duration,
                motion_frames=motion_frames,
                avatar_meta=avatar_meta,
                scene=scene,
            )
            ctx.set_artifact(sid, "clip", clip_path)
            logger.info("avatar_animate scene %d: clip rendered", sid)

            ctx.update_progress("avatar_animate", i + 1, len(scenes))
            if ctx.should_checkpoint_scene(i + 1):
                checkpoint.save(ctx)

    def _animate_scene(
        self,
        ref_path: str,
        audio_path: str,
        output_path: str,
        duration: float,
        motion_frames: list[str],
        avatar_meta: dict,
        scene: dict,
    ) -> None:
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            logger.error("avatar_animate: cannot read reference %s", ref_path)
            return

        audio_energy = self._analyze_audio_energy(audio_path)
        fps = 24
        total_frames = int(duration * fps)

        landmarks = avatar_meta.get("landmarks", {})
        bbox = avatar_meta.get("bbox", [])
        use_motion = len(motion_frames) > 0

        from ...ffmpeg_util import run_ffmpeg, video_encoder_args

        tmp_dir = os.path.join(os.path.dirname(output_path), f"_tmp_animate_{os.path.basename(output_path)}")
        os.makedirs(tmp_dir, exist_ok=True)

        mouth_region = self._estimate_mouth_region(ref_img, landmarks, bbox)

        for frame_idx in range(total_frames):
            t = frame_idx / fps

            frame = ref_img.copy()

            if use_motion:
                motion_idx = min(int(t * 2) % len(motion_frames), len(motion_frames) - 1)
                motion_img = cv2.imread(motion_frames[motion_idx])
                if motion_img is not None:
                    frame = self._blend_motion(ref_img, motion_img, landmarks, alpha=0.15)

            energy = audio_energy[min(int(t * 30), len(audio_energy) - 1)] if audio_energy else 0.0

            if mouth_region:
                frame = self._animate_mouth(frame, mouth_region, energy, t)

            frame = self._apply_micro_movement(frame, t, frame_idx)

            frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(frame_path, frame)

        cmd = [
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%05d.png"),
            "-i", audio_path,
        ]
        cmd += video_encoder_args()
        cmd += [
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-y", output_path,
        ]
        try:
            run_ffmpeg(cmd)
            logger.info("avatar_animate: ffmpeg encode done → %s", output_path)
        except Exception as exc:
            logger.error("avatar_animate: ffmpeg failed: %s", exc)
        finally:
            import shutil
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def _analyze_audio_energy(self, audio_path: str) -> list[float]:
        try:
            with wave.open(audio_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            if sampwidth == 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            elif sampwidth == 4:
                samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
            else:
                samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0

            if n_channels > 1:
                samples = samples[::n_channels]

            window_size = int(framerate / 30)
            hop = window_size
            energy = []
            for start in range(0, len(samples) - window_size, hop):
                chunk = samples[start : start + window_size]
                e = float(np.sqrt(np.mean(chunk ** 2)))
                energy.append(e)

            if energy:
                max_e = max(energy)
                if max_e > 0:
                    energy = [e / max_e for e in energy]

            return energy

        except Exception as exc:
            logger.warning("avatar_animate: audio analysis failed: %s", exc)
            return []

    def _estimate_mouth_region(
        self, img: np.ndarray, landmarks: dict, bbox: list,
    ) -> dict | None:
        h, w = img.shape[:2]

        if landmarks.get("mouth_left") and landmarks.get("mouth_right"):
            ml = landmarks["mouth_left"]
            mr = landmarks["mouth_right"]
            mc = landmarks.get("mouth_center", [(ml[0] + mr[0]) // 2, (ml[1] + mr[1]) // 2])
            mouth_w = abs(mr[0] - ml[0])
            mouth_h = int(mouth_w * 0.5)
            pad_x = int(mouth_w * 0.3)
            pad_y = int(mouth_h * 0.5)
            return {
                "x1": max(0, ml[0] - pad_x),
                "y1": max(0, mc[1] - mouth_h - pad_y),
                "x2": min(w, mr[0] + pad_x),
                "y2": min(h, mc[1] + mouth_h + pad_y),
                "center_x": mc[0],
                "center_y": mc[1],
                "mouth_width": mouth_w,
            }

        if len(bbox) >= 4:
            fx, fy, fw, fh = bbox
            scale = min(w / 512, h / 512)
            sfx, sfy, sfw, sfh = int(fx * scale), int(fy * scale), int(fw * scale), int(fh * scale)
            mouth_y = sfy + int(sfh * 0.72)
            mouth_x1 = sfx + int(sfw * 0.25)
            mouth_x2 = sfx + int(sfw * 0.75)
            mouth_w = mouth_x2 - mouth_x1
            mouth_h = int(mouth_w * 0.4)
            return {
                "x1": max(0, mouth_x1 - 10),
                "y1": max(0, mouth_y - mouth_h - 5),
                "x2": min(w, mouth_x2 + 10),
                "y2": min(h, mouth_y + mouth_h + 5),
                "center_x": (mouth_x1 + mouth_x2) // 2,
                "center_y": mouth_y,
                "mouth_width": mouth_w,
            }

        return None

    def _animate_mouth(self, frame: np.ndarray, mouth: dict, energy: float, t: float) -> np.ndarray:
        if energy < 0.05:
            return frame

        x1, y1 = int(mouth["x1"]), int(mouth["y1"])
        x2, y2 = int(mouth["x2"]), int(mouth["y2"])
        cx, cy = int(mouth["center_x"]), int(mouth["center_y"])
        mw = mouth["mouth_width"]

        if x2 <= x1 or y2 <= y1:
            return frame

        mouth_region = frame[y1:y2, x1:x2].copy()
        if mouth_region.size == 0:
            return frame

        open_amount = energy * 0.4
        pulse = math.sin(t * 12) * 0.15 * energy
        total_open = open_amount + pulse

        new_h = max(4, int(mouth_region.shape[0] * (1.0 + total_open)))
        new_w = int(mouth_region.shape[1] * (1.0 - total_open * 0.1))

        resized = cv2.resize(mouth_region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        paste_y = max(0, cy - new_h // 2)
        paste_x = max(0, cx - new_w // 2)
        paste_y_end = min(frame.shape[0], paste_y + new_h)
        paste_x_end = min(frame.shape[1], paste_x + new_w)

        src_y_end = paste_y_end - paste_y
        src_x_end = paste_x_end - paste_x

        if src_y_end > 0 and src_x_end > 0:
            blend = 0.7
            roi = frame[paste_y:paste_y_end, paste_x:paste_x_end]
            blended = cv2.addWeighted(resized[:src_y_end, :src_x_end], blend, roi, 1 - blend, 0)
            frame[paste_y:paste_y_end, paste_x:paste_x_end] = blended

        return frame

    def _apply_micro_movement(self, frame: np.ndarray, t: float, frame_idx: int) -> np.ndarray:
        dx = int(math.sin(t * 0.5) * 1.5)
        dy = int(math.cos(t * 0.3) * 1.0)

        if dx == 0 and dy == 0:
            return frame

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                               borderMode=cv2.BORDER_REPLICATE)
        return frame

    def _blend_motion(self, ref: np.ndarray, motion: np.ndarray, landmarks: dict, alpha: float) -> np.ndarray:
        if ref.shape != motion.shape:
            motion = cv2.resize(motion, (ref.shape[1], ref.shape[0]))

        blended = cv2.addWeighted(ref, 1 - alpha, motion, alpha, 0)
        return blended
