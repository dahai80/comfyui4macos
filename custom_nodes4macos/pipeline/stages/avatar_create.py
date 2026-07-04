from __future__ import annotations

import json
import logging
import os
import time

import cv2
import numpy as np

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.avatar_create")

_HAARCASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_LBPCASCADE_PROFILE = cv2.data.lbmcascades + "lbpcascade_profileface.xml" if hasattr(cv2.data, "lbmcascades") else ""


class AvatarCreateStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="avatar_create",
            description="从照片+视频提取面部特征，生成数字人形象包",
            model_requirements=[],
            memory_estimate_gb=1.0,
            input_kinds=["photo", "video"],
            output_kinds=["avatar_package"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        avatar_dir = os.path.join(ctx.job_dir, "_avatar")
        os.makedirs(avatar_dir, exist_ok=True)

        photo_path = ctx.config.get("avatar_photo", "")
        video_path = ctx.config.get("avatar_video", "")
        avatar_style = ctx.config.get("avatar_style", "realistic")

        logger.info(
            "avatar_create: photo=%s video=%s style=%s",
            photo_path, video_path, avatar_style,
        )

        face_data = {}
        reference_frame = None

        if photo_path and os.path.isfile(photo_path):
            face_data, reference_frame = self._process_photo(photo_path, avatar_dir)
        elif video_path and os.path.isfile(video_path):
            face_data, reference_frame = self._process_video(video_path, avatar_dir)

        if reference_frame is None:
            logger.warning("avatar_create: no face detected, generating placeholder")
            reference_frame = self._generate_placeholder(avatar_dir)

        if avatar_style == "cartoon":
            reference_frame = self._apply_cartoon_style(reference_frame, avatar_dir)

        ref_path = os.path.join(avatar_dir, "reference.png")
        cv2.imwrite(ref_path, reference_frame)
        logger.info("avatar_create: reference frame saved to %s", ref_path)

        face_data["avatar_style"] = avatar_style
        face_data["reference_frame"] = ref_path
        face_data["created_at"] = time.time()

        meta_path = os.path.join(avatar_dir, "avatar_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(face_data, f, ensure_ascii=False, indent=2)
        logger.info("avatar_create: metadata saved to %s", meta_path)

        ctx.config["avatar_package"] = avatar_dir
        ctx.config["avatar_reference"] = ref_path
        ctx.artifacts["avatar_package"] = avatar_dir
        ctx.artifacts["avatar_reference"] = ref_path

        logger.info("avatar_create: complete, style=%s", avatar_style)

    def _process_photo(self, photo_path: str, avatar_dir: str) -> tuple[dict, np.ndarray | None]:
        img = cv2.imread(photo_path)
        if img is None:
            logger.error("avatar_create: cannot read photo %s", photo_path)
            return {}, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_data = self._detect_face(gray, img.shape)
        if face_data:
            aligned = self._align_face(img, face_data)
            face_data["source"] = "photo"
            face_data["source_path"] = photo_path
            face_data["image_size"] = img.shape[:2]
            return face_data, aligned

        logger.warning("avatar_create: no face detected in photo %s", photo_path)
        return {}, img

    def _process_video(self, video_path: str, avatar_dir: str) -> tuple[dict, np.ndarray | None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("avatar_create: cannot open video %s", video_path)
            return {}, None

        best_face_data = {}
        best_frame = None
        best_score = 0
        frame_count = 0
        max_frames = 60

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 3 != 1:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_data = self._detect_face(gray, frame.shape)
            if face_data:
                score = face_data.get("confidence", 0) * face_data.get("face_area", 0)
                if score > best_score:
                    best_score = score
                    best_face_data = face_data
                    best_frame = frame.copy()

        cap.release()
        logger.info("avatar_create: scanned %d frames from video", frame_count)

        if best_frame is not None and best_face_data:
            aligned = self._align_face(best_frame, best_face_data)
            best_face_data["source"] = "video"
            best_face_data["source_path"] = video_path
            best_face_data["image_size"] = best_frame.shape[:2]

            self._extract_motion_frames(video_path, avatar_dir)
            return best_face_data, aligned

        return {}, best_frame

    def _detect_face(self, gray: np.ndarray, shape: tuple) -> dict:
        cascade = cv2.CascadeClassifier(_HAARCASCADE_PATH)
        faces = cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80),
            outputRejectLevels=True,
        )

        if len(faces[0]) == 0:
            logger.debug("avatar_create: no face detected")
            return {}

        rects = faces[0]
        reject_levels = faces[1]
        level_weights = faces[2] if len(faces) > 2 else None

        best_idx = 0
        if level_weights is not None and len(level_weights) > 0:
            best_idx = int(np.argmax(level_weights))

        x, y, w, h = rects[best_idx]
        confidence = float(level_weights[best_idx]) if level_weights is not None else 1.0

        landmarks = self._detect_landmarks(gray, x, y, w, h)

        h_img, w_img = shape[:2]
        return {
            "bbox": [int(x), int(y), int(w), int(h)],
            "confidence": round(confidence, 3),
            "face_area": int(w * h),
            "face_ratio": round(w / h, 3) if h > 0 else 0,
            "relative_position": [round(x / w_img, 3), round(y / h_img, 3)],
            "landmarks": landmarks,
        }

    def _detect_landmarks(self, gray: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> dict:
        landmarks = {}

        try:
            landmark_detector = cv2.FaceDetectorYN.create(
                "", "", (320, 320),
            )
        except Exception:
            pass

        eye_y = fy + int(fh * 0.35)
        eye_x_left = fx + int(fw * 0.3)
        eye_x_right = fx + int(fw * 0.7)
        nose_x = fx + int(fw * 0.5)
        nose_y = fy + int(fh * 0.6)
        mouth_y = fy + int(fh * 0.78)
        mouth_x_left = fx + int(fw * 0.35)
        mouth_x_right = fx + int(fw * 0.65)

        landmarks["left_eye"] = [eye_x_left, eye_y]
        landmarks["right_eye"] = [eye_x_right, eye_y]
        landmarks["nose_tip"] = [nose_x, nose_y]
        landmarks["mouth_left"] = [mouth_x_left, mouth_y]
        landmarks["mouth_right"] = [mouth_x_right, mouth_y]
        landmarks["mouth_center"] = [int((mouth_x_left + mouth_x_right) / 2), mouth_y]

        eye_dist = eye_x_right - eye_x_left
        landmarks["eye_distance"] = eye_dist
        landmarks["eye_angle"] = 0.0

        return landmarks

    def _align_face(self, img: np.ndarray, face_data: dict) -> np.ndarray:
        bbox = face_data.get("bbox", [])
        landmarks = face_data.get("landmarks", {})

        if len(bbox) < 4:
            return img

        x, y, w, h = bbox
        pad = int(max(w, h) * 0.4)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        face_crop = img[y1:y2, x1:x2]

        if face_crop.size == 0:
            return img

        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if left_eye and right_eye:
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            face_data["landmarks"]["eye_angle"] = round(angle, 2)

            if abs(angle) > 2.0:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                face_crop = cv2.warpAffine(
                    img, rot_mat, (img.shape[1], img.shape[0]),
                    flags=cv2.INTER_LINEAR,
                )
                face_crop = face_crop[y1:y2, x1:x2]

        target_size = 512
        face_crop = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return face_crop

    def _extract_motion_frames(self, video_path: str, avatar_dir: str) -> None:
        motion_dir = os.path.join(avatar_dir, "motion_frames")
        os.makedirs(motion_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // 8)
        saved = 0

        for idx in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            path = os.path.join(motion_dir, f"frame_{saved:03d}.png")
            cv2.imwrite(path, frame)
            saved += 1
            if saved >= 8:
                break

        cap.release()
        logger.info("avatar_create: extracted %d motion frames", saved)

    def _apply_cartoon_style(self, img: np.ndarray, avatar_dir: str) -> np.ndarray:
        logger.info("avatar_create: applying cartoon style via edge-preserving filter")

        small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        for _ in range(3):
            small = cv2.bilateralFilter(small, d=9, sigmaColor=75, sigmaSpace=75)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9,
        )
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(small, edges_colored)

        saturated = cartoon.astype(np.float32)
        saturated[:, :, 1] *= 1.1
        saturated[:, :, 2] *= 1.05
        saturated = np.clip(saturated, 0, 255).astype(np.uint8)

        result = cv2.resize(saturated, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        return result

    @staticmethod
    def _generate_placeholder(avatar_dir: str) -> np.ndarray:
        img = np.full((512, 512, 3), (40, 40, 60), dtype=np.uint8)
        cv2.ellipse(img, (256, 180), (80, 100), 0, 0, 360, (180, 180, 200), -1)
        cv2.rectangle(img, (206, 290), (306, 450), (100, 100, 140), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Avatar", (185, 490), font, 0.8, (200, 200, 220), 2)
        return img
