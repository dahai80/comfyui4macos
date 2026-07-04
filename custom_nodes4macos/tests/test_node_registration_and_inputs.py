"""补充覆盖：节点注册、节点 INPUT_TYPES、节点 fallback 路径。

覆盖 __init__.py 的 6 个 try/except 注册块、各节点的
list_models_safe 调用、_output_directory fallback、_bytes_to_image_tensor 等。
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestNodeRegistration(unittest.TestCase):
    """覆盖 custom_nodes4macos/__init__.py 的 6 个节点注册块。"""

    def test_all_six_nodes_registered(self):
        import custom_nodes4macos as pkg
        self.assertIn("FusionMLXPromptExpand", pkg.NODE_CLASS_MAPPINGS)
        self.assertIn("FusionMLXFluxImage", pkg.NODE_CLASS_MAPPINGS)
        self.assertIn("FusionMLXHorrorTTS", pkg.NODE_CLASS_MAPPINGS)
        self.assertIn("FusionMLXKenBurns", pkg.NODE_CLASS_MAPPINGS)
        self.assertIn("FusionMLXAssemble", pkg.NODE_CLASS_MAPPINGS)
        self.assertIn("FusionMLXDreamFactory", pkg.NODE_CLASS_MAPPINGS)

    def test_display_names_registered(self):
        import custom_nodes4macos as pkg
        self.assertEqual(len(pkg.NODE_DISPLAY_NAME_MAPPINGS), 6)
        for name in pkg.NODE_CLASS_MAPPINGS:
            self.assertIn(name, pkg.NODE_DISPLAY_NAME_MAPPINGS)

    def test_web_directory_set(self):
        import custom_nodes4macos as pkg
        self.assertEqual(pkg.WEB_DIRECTORY, "./web")

    def test_node_classes_are_correct_types(self):
        import custom_nodes4macos as pkg
        from custom_nodes4macos.nodes.prompt_expand import FusionMLXPromptExpand
        from custom_nodes4macos.nodes.flux_image import FusionMLXFluxImage
        self.assertIs(pkg.NODE_CLASS_MAPPINGS["FusionMLXPromptExpand"], FusionMLXPromptExpand)
        self.assertIs(pkg.NODE_CLASS_MAPPINGS["FusionMLXFluxImage"], FusionMLXFluxImage)


class TestFluxImageNodeFull(unittest.TestCase):
    """覆盖 nodes/flux_image.py 的未测路径。"""

    def test_input_types_structure(self):
        from custom_nodes4macos.nodes.flux_image import FusionMLXFluxImage
        inputs = FusionMLXFluxImage.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)
        req = inputs["required"]
        for key in ["visual_prompt", "global_style", "model", "width", "height", "steps", "guidance", "seed"]:
            self.assertIn(key, req)
        self.assertIn("base_url", inputs["optional"])
        self.assertIn("api_key", inputs["optional"])

    def test_bytes_to_image_tensor_shape(self):
        from custom_nodes4macos.nodes import flux_image as fi
        # 构造一个最小 PNG
        from PIL import Image
        import io
        img = Image.new("RGB", (4, 4), (10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        tensor = fi._bytes_to_image_tensor(buf.getvalue())
        self.assertEqual(tensor.shape[0], 1)  # batch dim
        self.assertEqual(tensor.shape[-1], 3)  # channels

    def test_bytes_to_image_tensor_values_normalized(self):
        from custom_nodes4macos.nodes import flux_image as fi
        from PIL import Image
        import io
        import torch
        img = Image.new("RGB", (2, 2), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        tensor = fi._bytes_to_image_tensor(buf.getvalue())
        # 红通道应接近 1.0
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1.0, places=1)

    def test_class_attributes(self):
        from custom_nodes4macos.nodes.flux_image import FusionMLXFluxImage
        self.assertEqual(FusionMLXFluxImage.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(FusionMLXFluxImage.RETURN_NAMES, ("image",))
        self.assertEqual(FusionMLXFluxImage.FUNCTION, "generate")
        self.assertEqual(FusionMLXFluxImage.CATEGORY, "FusionMLX/Horror")


class TestHorrorTtsNodeFull(unittest.TestCase):

    def test_input_types_structure(self):
        from custom_nodes4macos.nodes.horror_tts import FusionMLXHorrorTTS
        inputs = FusionMLXHorrorTTS.INPUT_TYPES()
        req = inputs["required"]
        for key in ["audio_script", "voice", "instructions", "model", "speed", "response_format", "filename_prefix"]:
            self.assertIn(key, req)

    def test_audio_extensions_constant(self):
        from custom_nodes4macos.nodes import horror_tts as ht
        self.assertEqual(ht._AUDIO_EXTS, ("wav",))

    def test_save_audio_creates_file(self):
        from custom_nodes4macos.nodes import horror_tts as ht
        with tempfile.TemporaryDirectory() as td:
            with patch.object(ht, "_output_directory", return_value=td):
                path = ht._save_audio(b"fake-audio", "prefix", "wav")
                self.assertTrue(os.path.exists(path))
                self.assertTrue(path.startswith(td))
                self.assertTrue(path.endswith(".wav"))
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), b"fake-audio")

    def test_save_audio_invalid_ext_defaults_wav(self):
        from custom_nodes4macos.nodes import horror_tts as ht
        with tempfile.TemporaryDirectory() as td:
            with patch.object(ht, "_output_directory", return_value=td):
                path = ht._save_audio(b"x", "p", "mp3")  # 非 wav
                self.assertTrue(path.endswith(".wav"))

    def test_class_attributes(self):
        from custom_nodes4macos.nodes.horror_tts import FusionMLXHorrorTTS
        self.assertEqual(FusionMLXHorrorTTS.RETURN_TYPES, ("STRING",))
        self.assertEqual(FusionMLXHorrorTTS.CATEGORY, "FusionMLX/Horror")


class TestKenBurnsNodeFull(unittest.TestCase):

    def test_input_types_structure(self):
        from custom_nodes4macos.nodes.ken_burns import FusionMLXKenBurns
        inputs = FusionMLXKenBurns.INPUT_TYPES()
        req = inputs["required"]
        for key in ["image", "duration_seconds", "motion_preset", "width", "height", "fps", "filename_prefix"]:
            self.assertIn(key, req)
        self.assertIn("audio_path", inputs["optional"])

    def test_presets_constant(self):
        from custom_nodes4macos.nodes import ken_burns as kb
        self.assertEqual(kb._PRESETS, ["zoom-in", "zoom-out", "pan-right", "pan-left", "random"])

    def test_image_tensor_to_png_handles_4d(self):
        from custom_nodes4macos.nodes import ken_burns as kb
        import numpy
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            path = tf.name
        try:
            arr = numpy.zeros((1, 4, 4, 3), dtype=numpy.uint8)
            kb._image_tensor_to_png(arr, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_image_tensor_to_png_handles_3d(self):
        from custom_nodes4macos.nodes import ken_burns as kb
        import numpy
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            path = tf.name
        try:
            arr = numpy.zeros((4, 4, 3), dtype=numpy.uint8)
            kb._image_tensor_to_png(arr, path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_image_tensor_to_png_normalizes_float(self):
        from custom_nodes4macos.nodes import ken_burns as kb
        import numpy
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            path = tf.name
        try:
            arr = numpy.ones((2, 2, 3), dtype=numpy.float32)  # 1.0 应转 255
            kb._image_tensor_to_png(arr, path)
            from PIL import Image
            img = Image.open(path)
            self.assertEqual(img.getpixel((0, 0)), (255, 255, 255))
        finally:
            os.unlink(path)

    def test_image_tensor_to_png_invalid_shape_raises(self):
        from custom_nodes4macos.nodes import ken_burns as kb
        import numpy
        arr = numpy.zeros((5,), dtype=numpy.uint8)  # 1D，不支持
        with self.assertRaises(ValueError):
            kb._image_tensor_to_png(arr, "/tmp/x.png")


class TestAssembleNodeFull(unittest.TestCase):

    def test_input_types_structure(self):
        from custom_nodes4macos.nodes.assemble import FusionMLXAssemble
        inputs = FusionMLXAssemble.INPUT_TYPES()
        req = inputs["required"]
        for key in ["clips", "transition", "width", "height", "fps", "filename_prefix"]:
            self.assertIn(key, req)

    def test_transitions_constant(self):
        from custom_nodes4macos.nodes import assemble as asm
        self.assertEqual(asm._TRANSITIONS, ["none", "fade"])

    def test_bgm_volume_constant(self):
        from custom_nodes4macos.nodes import assemble as asm
        self.assertEqual(asm._BGM_VOLUME, 0.3)

    def test_parse_clips_strips_quotes(self):
        from custom_nodes4macos.nodes import assemble as asm
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            path = tf.name
        try:
            text = f'"{path}"'  # 带双引号
            clips = asm._parse_clips(text)
            self.assertEqual(clips, [path])
        finally:
            os.unlink(path)

    def test_parse_clips_strips_single_quotes(self):
        from custom_nodes4macos.nodes import assemble as asm
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            path = tf.name
        try:
            text = f"'{path}'"
            clips = asm._parse_clips(text)
            self.assertEqual(clips, [path])
        finally:
            os.unlink(path)

    def test_class_attributes(self):
        from custom_nodes4macos.nodes.assemble import FusionMLXAssemble
        self.assertEqual(FusionMLXAssemble.CATEGORY, "FusionMLX/Horror")
        self.assertEqual(FusionMLXAssemble.FUNCTION, "render")


class TestOutputDirectoryFallback(unittest.TestCase):
    """覆盖 _output_directory 的 folder_paths 不可用 fallback。"""

    def test_flux_image_no_folder_paths_uses_tempfile(self):
        # folder_paths 模块不存在时各节点 fallback 到 tempfile
        from custom_nodes4macos.nodes import ken_burns as kb
        result = kb._output_directory()
        self.assertIsInstance(result, str)
        self.assertTrue(os.path.isdir(result))


if __name__ == "__main__":
    unittest.main()
