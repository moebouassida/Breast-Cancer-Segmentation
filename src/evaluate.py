"""
evaluate.py — Standalone evaluation + quality gate.

Usage:
    python src/evaluate.py
    python src/evaluate.py --checkpoint checkpoints/best.pt
    python src/evaluate.py --checkpoint checkpoints/best.pt --split test

Exit codes:
    0 — passed all quality gates
    1 — failed one or more quality gates
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import UNet
from src.metrics import compute_all_metrics


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model = UNet(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    state = ckpt.get("model_state", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    best = ckpt.get("best_val", "?")
    if isinstance(best, float):
        print(f"  Checkpoint: epoch={epoch}, best_val_dice={best:.4f}")
    return model


def get_loader(cfg, split: str):
    try:
        from Data.data_loader import get_dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(cfg)
        return {"train": train_loader, "val": val_loader, "test": test_loader}[split]
    except Exception as e:
        print(f"⚠️  Could not load real data ({e}). Using synthetic data for CI.")
        return _synthetic_loader(cfg)


def _synthetic_loader(cfg):
    class Syn(torch.utils.data.Dataset):
        def __len__(self): return 32
        def __getitem__(self, _):
            img = torch.rand(1, cfg.img_size, cfg.img_size)
            mask = (torch.rand(1, cfg.img_size, cfg.img_size) > 0.5).float()
            return img, mask
    return torch.utils.data.DataLoader(Syn(), batch_size=cfg.batch_size)


def evaluate(checkpoint_path: str, split: str = "test", output_json: str = None) -> bool:
    cfg = Config()
    device = cfg.device

    print(f"\n{'='*58}")
    print(f"  Breast Ultrasound Segmentation — Evaluation")
    print(f"{'='*58}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Split      : {split}")
    print(f"  Device     : {device}")
    print(f"{'='*58}\n")

    model = load_model(checkpoint_path, device)
    loader = get_loader(cfg, split)

    accumulated = {k: [] for k in ["dice", "iou", "pixel_accuracy", "precision", "recall", "f1"]}

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()
            batch = compute_all_metrics(preds, masks)
            for k, v in batch.items():
                accumulated[k].append(v)

    metrics = {k: round(sum(v) / len(v), 4) for k, v in accumulated.items()}

    # Quality gates from config
    gates = {
        "dice":      cfg.gate_dice,
        "iou":       cfg.gate_iou,
        "precision": cfg.gate_precision,
        "recall":    cfg.gate_recall,
    }

    # Print results table
    print(f"  {'Metric':<20} {'Value':>8}   {'Threshold':>10}   {'Status'}")
    print(f"  {'-'*54}")

    passed_all = True
    for metric, threshold in gates.items():
        value = metrics[metric]
        passed = value >= threshold
        if not passed:
            passed_all = False
        status = "✅ PASS" if passed else "❌ FAIL"
        note = "  ← BELOW THRESHOLD" if not passed else ""
        print(f"  {metric:<20} {value:>8.4f}   {threshold:>10.4f}   {status}{note}")

    print(f"  {'pixel_accuracy':<20} {metrics['pixel_accuracy']:>8.4f}   {'(no gate)':>10}")
    print(f"  {'f1':<20} {metrics['f1']:>8.4f}   {'(no gate)':>10}")
    print(f"\n  Batches: {len(accumulated['dice'])}")
    print(f"{'='*58}")

    if passed_all:
        print("\n  ✅ All quality gates passed — ready for deployment.\n")
    else:
        print("\n  ❌ Quality gate failed — blocking deployment.\n")

    # Save results
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({
                "metrics": metrics,
                "gates": gates,
                "passed": passed_all,
                "checkpoint": checkpoint_path,
                "split": split,
            }, f, indent=2)
        print(f"  Results saved → {output_json}")

    return passed_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-json", default="metrics/eval_results.json")
    args = parser.parse_args()

    passed = evaluate(args.checkpoint, args.split, args.output_json)
    sys.exit(0 if passed else 1)