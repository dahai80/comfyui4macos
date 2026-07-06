from __future__ import annotations

import json
import logging
import os
import subprocess
import time

import cv2
import numpy as np

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.avatar_create")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types (int32, float32, etc.)."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Phase 2 optimizations:
#   C1: Extensible face detection — tries MediaPipe (tasks API) first,
#       falls back to OpenCV Haar Cascade.  MediaPipe model can be
#       placed at ~/.cache/mediapipe/models/face_detection_short_range.tflite
#   C2: ffmpeg-based keyframe extraction — replaces cv2.VideoCapture seek


class AvatarCreateStage(Stage):

    _MEDIAPIPE_MODEL_PATH = os.path.expanduser(
        "~/.cache/mediapipe/models/face_landmarker.task"
    )
    _HAARCASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    # Lazy-loaded face detector (None = uninitialized, False = MediaPipe unavailable)
    _mp_face_detector = None

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

    @classmethod
    def _get_mediapipe_detector(cls):
        """C1: Lazy-init MediaPipe Face Detection (tasks API).

        Requires the model file at _MEDIAPIPE_MODEL_PATH.  If the model
        file is absent, returns None and the caller falls back to Haar.
        """
        if cls._mp_face_detector is None:
            if not os.path.isfile(cls._MEDIAPIPE_MODEL_PATH):
                logger.info(
                    "[avatar_create] MediaPipe model not found at %s, "
                    "using Haar Cascade (install model for 10-20x speedup and precise landmarks)",
                    cls._MEDIAPIPE_MODEL_PATH,
                )
                cls._mp_face_detector = False  # Don't retry
                return None

            try:
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python import BaseOptions

                options = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=cls._MEDIAPIPE_MODEL_PATH,
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                )
                cls._mp_face_detector = vision.FaceLandmarker.create_from_options(options)
                logger.info("[avatar_create] MediaPipe Face Landmarker initialized")
            except Exception as exc:
                logger.warning("[avatar_create] MediaPipe init failed: %s", exc)
                cls._mp_face_detector = False

        return cls._mp_face_detector if cls._mp_face_detector is not False else None

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        pre_pkg = ctx.config.get("avatar_package", "") or ctx.artifacts.get("avatar_package", "")
        if pre_pkg and os.path.isdir(pre_pkg):
            ref_in_pkg = os.path.join(pre_pkg, "reference.png")
            meta_in_pkg = os.path.join(pre_pkg, "avatar_meta.json")
            if os.path.isfile(ref_in_pkg) and os.path.isfile(meta_in_pkg):
                ctx.artifacts["avatar_package"] = pre_pkg
                ctx.artifacts["avatar_reference"] = ref_in_pkg
                ctx.config["avatar_reference"] = ref_in_pkg
                logger.info(
                    "avatar_create: reuse pre-built avatar_package=%s, skip face detection",
                    pre_pkg,
                )
                return
            logger.warning(
                "avatar_create: avatar_package=%s missing reference.png/avatar_meta.json, rebuild",
                pre_pkg,
            )

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
            json.dump(face_data, f, cls=_NumpyEncoder, ensure_ascii=False, indent=2)
        logger.info("avatar_create: metadata saved to %s", meta_path)

        ctx.config["avatar_package"] = avatar_dir
        ctx.config["avatar_reference"] = ref_path
        ctx.artifacts["avatar_package"] = avatar_dir
        ctx.artifacts["avatar_reference"] = ref_path

        logger.info("avatar_create: complete, style=%s", avatar_style)

    # ------------------------------------------------------------------
    # C1: MediaPipe-based face detection
    # ------------------------------------------------------------------

    def _detect_face_mediapipe(self, rgb_img: np.ndarray, shape: tuple) -> dict:
        """C1: Detect face using MediaPipe FaceLandmarker.

        Provides bounding box + 478 face landmarks including precise mouth/eye/nose.
        Returns face_data dict with bbox and landmarks, or {} if no face found.
        10-20x faster than Haar Cascade on Apple Silicon.
        """
        landmarker = self._get_mediapipe_detector()
        if landmarker is None:
            # Fallback to original Haar Cascade
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            return self._detect_face_haar(gray, shape)

        try:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            result = landmarker.detect(mp_image)

            if not result or not result.face_landmarks:
                return {}

            landmarks_list = result.face_landmarks[0]
            h_img, w_img = shape[:2]

            # Compute bounding box from all landmarks
            xs = [lm.x * w_img for lm in landmarks_list]
            ys = [lm.y * h_img for lm in landmarks_list]
            x, y = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            w, h = x2 - x, y2 - y

            # Extract key facial landmarks (MediaPipe FaceMesh indices)
            # Left eye: 33, 133 | Right eye: 362, 263 | Nose tip: 1
            # Mouth corners: 61, 291 | Mouth center: 13
            def _pt(idx):
                lm = landmarks_list[idx]
                return [int(lm.x * w_img), int(lm.y * h_img)]

            landmarks = {
                "left_eye": _pt(133),
                "right_eye": _pt(362),
                "nose_tip": _pt(1),
                "mouth_left": _pt(61),
                "mouth_right": _pt(291),
                "mouth_center": _pt(13),
                "eye_distance": int(abs(_pt(362)[0] - _pt(133)[0])),
                "eye_angle": 0.0,
            }

            confidence = 0.95  # MediaPipe FaceLandmarker is highly reliable

            return {
                "bbox": [x, y, w, h],
                "confidence": confidence,
                "face_area": w * h,
                "face_ratio": round(w / h, 3) if h > 0 else 0,
                "relative_position": [round(x / w_img, 3), round(y / h_img, 3)],
                "landmarks": landmarks,
            }

        except Exception as exc:
            logger.warning("avatar_create: MediaPipe detection failed: %s", exc)
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            return self._detect_face_haar(gray, shape)

    # ------------------------------------------------------------------
    # Legacy Haar Cascade fallback (preserved for environments without MediaPipe)
    # ------------------------------------------------------------------

    _HAARCASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def _detect_face_haar(self, gray: np.ndarray, shape: tuple) -> dict:
        """Original Haar Cascade face detection (fallback when MediaPipe unavailable)."""
        if not hasattr(cv2, "CascadeClassifier"):
            if not getattr(self, "_haar_unavailable_warned", False):
                logger.warning(
                    "avatar_create: cv2.CascadeClassifier unavailable (OpenCV %s), "
                    "haar fallback disabled — returning no-face",
                    cv2.__version__,
                )
                self._haar_unavailable_warned = True
            return {}
        cascade = cv2.CascadeClassifier(self._HAARCASCADE_PATH)
        faces = cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80),
            outputRejectLevels=True,
        )

        if len(faces[0]) == 0:
            return {}

        rects = faces[0]
        level_weights = faces[2] if len(faces) > 2 else None

        best_idx = 0
        if level_weights is not None and len(level_weights) > 0:
            best_idx = int(np.argmax(level_weights))

        x, y, w, h = rects[best_idx]
        confidence = float(level_weights[best_idx]) if level_weights is not None else 1.0

        landmarks = self._detect_landmarks_geometric(gray, x, y, w, h)

        h_img, w_img = shape[:2]
        return {
            "bbox": [int(x), int(y), int(w), int(h)],
            "confidence": round(confidence, 3),
            "face_area": int(w * h),
            "face_ratio": round(w / h, 3) if h > 0 else 0,
            "relative_position": [round(x / w_img, 3), round(y / h_img, 3)],
            "landmarks": landmarks,
        }

    def _detect_face(self, gray: np.ndarray, shape: tuple) -> dict:
        """Unified entry point — tries MediaPipe first, falls back to Haar."""
        # Convert gray back to RGB for MediaPipe
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        result = self._detect_face_mediapipe(rgb, shape)
        if result:
            return result
        return self._detect_face_haar(gray, shape)

    # ------------------------------------------------------------------
    # C1 improved landmarks from geometric estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_landmarks_geometric(gray: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> dict:
        """Geometric landmark estimation (Haar fallback)."""
        landmarks = {}
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

    # ------------------------------------------------------------------
    # Photo / video processing
    # ------------------------------------------------------------------

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
        """C2: Process video using ffmpeg for fast frame extraction + MediaPipe for detection."""

        # Use ffmpeg to extract keyframes at regular intervals
        frame_paths = self._extract_keyframes_ffmpeg(video_path, avatar_dir, max_frames=20)
        if not frame_paths:
            logger.warning("avatar_create: could not extract frames from video, falling back")
            cap = cv2.VideoCapture(video_path)
            return self._process_video_cv(cap, video_path, avatar_dir)

        best_face_data = {}
        best_frame = None
        best_score = 0

        for fp in frame_paths:
            frame = cv2.imread(fp)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_data = self._detect_face(gray, frame.shape)
            if face_data:
                score = face_data.get("confidence", 0) * face_data.get("face_area", 0)
                if score > best_score:
                    best_score = score
                    best_face_data = face_data
                    best_frame = frame.copy()

        # Clean up temp frames
        for fp in frame_paths:
            try:
                os.remove(fp)
            except OSError:
                pass

        if best_frame is not None and best_face_data:
            aligned = self._align_face(best_frame, best_face_data)
            best_face_data["source"] = "video"
            best_face_data["source_path"] = video_path
            best_face_data["image_size"] = best_frame.shape[:2]

            self._extract_motion_frames_ffmpeg(video_path, avatar_dir)
            return best_face_data, aligned

        return {}, best_frame

    def _process_video_cv(self, cap, video_path: str, avatar_dir: str) -> tuple[dict, np.ndarray | None]:
        """Original cv2-based video processing (fallback)."""
        import gc
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
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
        gc.collect()
        logger.info("avatar_create: scanned %d frames from video (cv fallback)", frame_count)

        if best_frame is not None and best_face_data:
            aligned = self._align_face(best_frame, best_face_data)
            best_face_data["source"] = "video"
            best_face_data["source_path"] = video_path
            best_face_data["image_size"] = best_frame.shape[:2]

            self._extract_motion_frames_ffmpeg(video_path, avatar_dir)
            return best_face_data, aligned

        return {}, best_frame

    # ------------------------------------------------------------------
    # C2: ffmpeg-based keyframe extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keyframes_ffmpeg(
        video_path: str, avatar_dir: str, max_frames: int = 20,
    ) -> list[str]:
        """C2: Extract evenly-spaced frames using ffmpeg (fast seeking).

        Much faster than cv2.VideoCapture seek-on-each-frame.
        Returns list of extracted frame paths.
        """
        import subprocess as sp

        # Get video duration via ffprobe
        try:
            dur_result = sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=15,
            )
            duration = float(dur_result.stdout.strip())
        except Exception:
            logger.warning("avatar_create: ffprobe failed, using cv fallback")
            return []

        frame_paths = []
        tmp_dir = os.path.join(avatar_dir, "_tmp_frames")
        os.makedirs(tmp_dir, exist_ok=True)

        interval = duration / max_frames
        for i in range(max_frames):
            t = i * interval + interval * 0.1  # Offset slightly from exact boundaries
            out_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
            try:
                sp.run(
                    ["ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                     "-vframes", "1", "-q:v", "3", out_path],
                    capture_output=True, timeout=30,
                )
                if os.path.isfile(out_path) and os.path.getsize(out_path) > 100:
                    frame_paths.append(out_path)
                else:
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
            except Exception:
                continue

        logger.info("avatar_create: ffmpeg extracted %d/%d frames", len(frame_paths), max_frames)
        return frame_paths

    @staticmethod
    def _extract_motion_frames_ffmpeg(video_path: str, avatar_dir: str, count: int = 8) -> None:
        """C2: Extract evenly-spaced motion frames using ffmpeg (replaces cv2 seek)."""
        import subprocess as sp

        motion_dir = os.path.join(avatar_dir, "motion_frames")
        os.makedirs(motion_dir, exist_ok=True)

        try:
            dur_result = sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=15,
            )
            duration = float(dur_result.stdout.strip())
        except Exception:
            logger.warning("avatar_create: ffprobe failed for motion frames")
            return

        if duration <= 0:
            return

        saved = 0
        for i in range(count):
            t = duration * (i + 0.5) / count
            out_path = os.path.join(motion_dir, f"frame_{saved:03d}.png")
            try:
                sp.run(
                    ["ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                     "-vframes", "1", "-q:v", "2", out_path],
                    capture_output=True, timeout=30,
                )
                if os.path.isfile(out_path) and os.path.getsize(out_path) > 100:
                    saved += 1
            except Exception:
                continue

        logger.info("avatar_create: ffmpeg extracted %d motion frames", saved)

    # Legacy wrapper (preserve API compatibility)
    def _extract_motion_frames(self, video_path: str, avatar_dir: str) -> None:
        self._extract_motion_frames_ffmpeg(video_path, avatar_dir)

    # ------------------------------------------------------------------
    # Face alignment (unchanged from original)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Cartoon style — anime-style: bright, saturated, clean black outlines
    # ------------------------------------------------------------------

    def _apply_cartoon_style(self, img: np.ndarray, avatar_dir: str) -> np.ndarray:
        """Anime-style cartoon avatar — vibrant, saturated, clean outlines.

        Pipeline:
          1. 3× saturation + 2.5× brightness on raw face (HSV)
          2. Light bilateral filter — smooth skin, keep edges
          3. Canny edge detection → thin black anime outlines
          4. Final brightness recovery boost
        """
        logger.info("avatar_create: applying anime cartoon style")

        # 1. Aggressive HSV boost on RAW face
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 3.0, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 2.5, 0, 255)
        boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 2. Light edge-preserving smooth (remove noise, keep color regions)
        smooth = cv2.bilateralFilter(boosted, 7, 40, 40)

        # 3. Edge detection — Canny on smooth image
        gray_s = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        gray_s = cv2.GaussianBlur(gray_s, (3, 3), 0)
        edges = cv2.Canny(gray_s, 40, 120)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 4. Anime-style black outlines — darken only edge pixels
        result = smooth.copy()
        result[edges > 0] = np.clip(
            result[edges > 0].astype(np.float32) * 0.3, 0, 255
        ).astype(np.uint8)

        # 5. Final brightness recovery
        hsv2 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv2[:, :, 2] = np.clip(hsv2[:, :, 2] * 1.2, 0, 255)
        result = cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return result

    @staticmethod
    def _generate_placeholder(avatar_dir: str) -> np.ndarray:
        img = np.full((512, 512, 3), (40, 40, 60), dtype=np.uint8)
        cv2.ellipse(img, (256, 180), (80, 100), 0, 0, 360, (180, 180, 200), -1)
        cv2.rectangle(img, (206, 290), (306, 450), (100, 100, 140), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Avatar", (185, 490), font, 0.8, (200, 200, 220), 2)
        return img
