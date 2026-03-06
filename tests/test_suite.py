"""
Comprehensive unit tests — run in CI without GPU or real data.
    pytest tests/ -v
"""

import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Model ─────────────────────────────────────────────────────────────────────
class TestUNet:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.model import UNet

        self.model = UNet(in_channels=1, out_channels=1).eval()

    def test_instantiation(self):
        from src.model import UNet

        assert UNet(in_channels=1, out_channels=1) is not None

    def test_output_shape_128(self):
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (1, 1, 128, 128)

    def test_output_shape_256(self):
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_batch_size_4(self):
        x = torch.randn(4, 1, 128, 128)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (4, 1, 128, 128)

    def test_no_nan_in_output(self):
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            out = self.model(x)
        assert not torch.isnan(out).any()

    def test_sigmoid_output_bounded(self):
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            out = torch.sigmoid(self.model(x))
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_gradients_flow(self):
        model = self.model.train()
        x = torch.randn(1, 1, 128, 128)
        y = torch.zeros(1, 1, 128, 128)
        loss = torch.nn.BCEWithLogitsLoss()(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient: {name}"

    def test_parameter_count_reasonable(self):
        from src.model import count_parameters

        n = count_parameters(self.model)
        assert 1_000_000 < n < 100_000_000, f"Unexpected param count: {n:,}"


# ── Metrics ───────────────────────────────────────────────────────────────────
class TestMetrics:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.metrics import (
            dice_score,
            iou_score,
            pixel_accuracy,
            precision_score,
            recall_score,
            f1_score,
            compute_all_metrics,
        )

        self.dice = dice_score
        self.iou = iou_score
        self.acc = pixel_accuracy
        self.prec = precision_score
        self.rec = recall_score
        self.f1 = f1_score
        self.all = compute_all_metrics

    def ones(self, shape=(1, 1, 64, 64)):
        return torch.ones(shape)

    def zeros(self, shape=(1, 1, 64, 64)):
        return torch.zeros(shape)

    def test_dice_perfect(self):
        m = self.ones()
        assert abs(self.dice(m, m).item() - 1.0) < 1e-4

    def test_dice_no_overlap(self):
        assert self.dice(self.zeros(), self.ones()).item() < 0.01

    def test_dice_range(self):
        for _ in range(20):
            p = (torch.rand(2, 1, 32, 32) > 0.5).float()
            t = (torch.rand(2, 1, 32, 32) > 0.5).float()
            assert 0.0 <= self.dice(p, t).item() <= 1.0

    def test_iou_perfect(self):
        m = self.ones()
        assert abs(self.iou(m, m).item() - 1.0) < 1e-4

    def test_iou_range(self):
        for _ in range(20):
            p = (torch.rand(2, 1, 32, 32) > 0.5).float()
            t = (torch.rand(2, 1, 32, 32) > 0.5).float()
            assert 0.0 <= self.iou(p, t).item() <= 1.0

    def test_dice_gte_iou(self):
        """Dice is always >= IoU for the same prediction."""
        for _ in range(20):
            p = (torch.rand(2, 1, 32, 32) > 0.5).float()
            t = (torch.rand(2, 1, 32, 32) > 0.5).float()
            assert self.dice(p, t).item() >= self.iou(p, t).item() - 1e-5

    def test_pixel_accuracy_perfect(self):
        m = self.ones()
        assert abs(self.acc(m, m).item() - 1.0) < 1e-4

    def test_pixel_accuracy_all_wrong(self):
        assert self.acc(self.zeros(), self.ones()).item() < 1e-4

    def test_precision_perfect(self):
        m = self.ones()
        assert abs(self.prec(m, m).item() - 1.0) < 1e-4

    def test_recall_perfect(self):
        m = self.ones()
        assert abs(self.rec(m, m).item() - 1.0) < 1e-4

    def test_f1_perfect(self):
        m = self.ones()
        assert abs(self.f1(m, m).item() - 1.0) < 1e-4

    def test_compute_all_metrics_keys(self):
        p = (torch.rand(2, 1, 32, 32) > 0.5).float()
        t = (torch.rand(2, 1, 32, 32) > 0.5).float()
        result = self.all(p, t)
        for key in ["dice", "iou", "pixel_accuracy", "precision", "recall", "f1"]:
            assert key in result
            assert 0.0 <= result[key] <= 1.0


# ── Config ────────────────────────────────────────────────────────────────────
class TestConfig:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.config import Config

        self.cfg = Config()

    def test_img_size_divisible_by_16(self):
        assert self.cfg.img_size % 16 == 0

    def test_learning_rate_sensible(self):
        assert 1e-6 < self.cfg.learning_rate < 1.0

    def test_batch_size_positive(self):
        assert self.cfg.batch_size > 0

    def test_epochs_positive(self):
        assert self.cfg.epochs > 0

    def test_device_valid(self):
        assert self.cfg.device in ("cuda", "cpu")

    def test_splits_sum_to_one(self):
        total = self.cfg.train_split + self.cfg.val_split
        assert total < 1.0  # remainder goes to test

    def test_quality_gates_in_range(self):
        for gate in [
            self.cfg.gate_dice,
            self.cfg.gate_iou,
            self.cfg.gate_precision,
            self.cfg.gate_recall,
        ]:
            assert 0.0 < gate < 1.0


# ── Preprocessing Pipeline ───────────────────────────────────────────────────
class TestPreprocessing:

    def test_transform_shape(self):
        import torchvision.transforms as T
        from PIL import Image

        transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
        img = Image.fromarray(
            np.random.randint(0, 255, (300, 200), dtype=np.uint8), mode="L"
        )
        t = transform(img)
        assert t.shape == (1, 128, 128)

    def test_transform_value_range(self):
        import torchvision.transforms as T
        from PIL import Image

        transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
        img = Image.fromarray(
            np.random.randint(0, 255, (128, 128), dtype=np.uint8), mode="L"
        )
        t = transform(img)
        assert 0.0 <= t.min().item() and t.max().item() <= 1.0

    def test_full_inference_pipeline(self):
        """transform → model → mask — no errors, correct shape."""
        import torchvision.transforms as T
        from PIL import Image
        from src.model import UNet

        transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
        model = UNet(in_channels=1, out_channels=1).eval()
        img = Image.fromarray(
            np.random.randint(0, 255, (300, 300), dtype=np.uint8), mode="L"
        )
        t = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = torch.sigmoid(model(t))
        assert out.shape == (1, 1, 128, 128)
        assert out.min().item() >= 0.0 and out.max().item() <= 1.0


# ── BUSI Dataset (skipped if data not present) ───────────────────────────────
class TestBUSIDataset:

    @pytest.mark.skipif(
        not os.path.exists("Data/Dataset_BUSI_with_GT"),
        reason="BUSI dataset not available",
    )
    def test_dataset_loads(self):
        from Data.data_loader import BUSIDataset
        from src.config import Config

        cfg = Config()
        ds = BUSIDataset(cfg.data_root, cfg.img_size)
        assert len(ds) > 0

    @pytest.mark.skipif(
        not os.path.exists("Data/Dataset_BUSI_with_GT"),
        reason="BUSI dataset not available",
    )
    def test_dataset_item_shapes(self):
        from Data.data_loader import BUSIDataset
        from src.config import Config

        cfg = Config()
        ds = BUSIDataset(cfg.data_root, cfg.img_size)
        img, mask = ds[0]
        assert img.shape == (1, cfg.img_size, cfg.img_size)
        assert mask.shape == (1, cfg.img_size, cfg.img_size)
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})
