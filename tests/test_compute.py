import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from traffic_edge.compute import compute_status, normalize_device, resolve_device
from traffic_edge import web_ui


class ComputeTests(unittest.TestCase):
    def torch(self, available):
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: available, get_device_name=lambda index: "Test GPU"),
            version=SimpleNamespace(cuda="12.6" if available else None),
            __version__="test",
        )

    def test_auto_falls_back_to_cpu_and_explicit_cuda_explains_missing_support(self):
        with patch.dict(sys.modules, {"torch": self.torch(False)}):
            self.assertEqual(resolve_device("auto"), "cpu")
            self.assertEqual(resolve_device("cpu"), "cpu")
            with self.assertRaisesRegex(ValueError, "NVIDIA CUDA is unavailable"):
                resolve_device("cuda")
            self.assertFalse(compute_status()["cuda_available"])

    def test_auto_and_explicit_cuda_select_first_gpu(self):
        with patch.dict(sys.modules, {"torch": self.torch(True)}):
            self.assertEqual(resolve_device("auto"), "cuda:0")
            self.assertEqual(resolve_device("cuda"), "cuda:0")
            self.assertEqual(resolve_device("cpu"), "cpu")
            self.assertEqual(compute_status()["gpu_name"], "Test GPU")

    def test_invalid_devices_are_rejected(self):
        for value in ("cuda:99", "mps", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_device(value)

    def test_yolo_receives_selected_device(self):
        from traffic_edge.backends import UltralyticsByteTrackBackend

        model = Mock()
        model.track.return_value = [SimpleNamespace(boxes=None)]
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model))}), \
                patch("traffic_edge.backends.resolve_device", return_value="cuda:0"):
            backend = UltralyticsByteTrackBackend("yolov8n.pt", .1, .5, [2, 5, 7], device="cuda")
            self.assertEqual(backend.track(None), [])
        self.assertEqual(model.track.call_args.kwargs["device"], "cuda:0")

    def test_web_configuration_retains_cpu_choice(self):
        with patch.object(web_ui.Path, "is_file", return_value=True):
            config = web_ui.create_config(
                "compute-test", [[.1, .1], [.9, .1], [.5, .9]], .25,
                direction=[[.5, .8], [.5, .2]],
                enabled_rules={"wrong_way": True}, device="cpu",
            )
        self.assertEqual(config["yolo"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
