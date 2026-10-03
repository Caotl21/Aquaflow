"""Behavior-cloning training entry point for hofa_policy."""
import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .actuation import ThrusterGeometry
from .contract import CONTRACT
from .dataset import PolicyDataset
from .model import HofaPolicyNet, ModelConfig


def overfit_indices(dataset_length, requested_samples):
    """Select a deterministic, run-wide subset for engineering overfit."""
    if dataset_length <= 0:
        raise ValueError("cannot select from an empty dataset")
    if requested_samples <= 0:
        raise ValueError("overfit sample count must be positive")
    count = min(dataset_length, requested_samples)
    if count == dataset_length:
        return list(range(dataset_length))
    return np.linspace(0, dataset_length - 1, count, dtype=np.int64).tolist()


def evaluate(model, loader, device):
    model.eval()
    squared = []
    absolute = []
    effort = []
    with torch.inference_mode():
        for state, reference, previous, target in loader:
            state = state.to(device)
            reference = reference.to(device)
            previous = previous.to(device)
            target = target.to(device)
            wrench, _, normalized = model(state, reference, previous)
            error = wrench - target
            squared.append(error.square().cpu())
            absolute.append(error.abs().cpu())
            effort.append(normalized.square().mean(dim=1).cpu())
    squared = torch.cat(squared)
    absolute = torch.cat(absolute)
    return {
        "rmse_fx": float(squared[:, 0].mean().sqrt()),
        "rmse_fy": float(squared[:, 1].mean().sqrt()),
        "rmse_nz": float(squared[:, 2].mean().sqrt()),
        "mae_fx": float(absolute[:, 0].mean()),
        "mae_fy": float(absolute[:, 1].mean()),
        "mae_nz": float(absolute[:, 2].mean()),
        "normalized_effort": float(torch.cat(effort).mean()),
    }


def train(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("output already exists: %s" % output)
    output.mkdir(parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_source = PolicyDataset(args.train)
    if args.mode == "overfit":
        selected = overfit_indices(len(train_source), args.overfit_samples)
        train_data = Subset(train_source, selected)
        val_data = Subset(train_source, selected)
        val_source = train_source
    else:
        if not args.val:
            raise ValueError("full training requires held-out validation runs")
        train_data = train_source
        val_source = PolicyDataset(args.val)
        val_data = val_source
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=generator)
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    geometry = ThrusterGeometry.bricsbot()
    config = ModelConfig()
    model = HofaPolicyNet(geometry, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run = {
        "contract": CONTRACT.to_dict(),
        "model_config": config.to_dict(),
        "device": str(device),
        "torch": str(torch.__version__),
        "seed": args.seed,
        "mode": args.mode,
        "overfit_samples": args.overfit_samples if args.mode == "overfit" else None,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_sources": train_source.sources,
        "val_sources": val_source.sources,
        "geometry": {
            "allocation_matrix": geometry.allocation_matrix.tolist(),
            "force_min": geometry.force_min.tolist(),
            "force_max": geometry.force_max.tolist(),
        },
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n")

    history = []
    best_score = float("inf")
    effort_weight = 0.0 if args.mode == "overfit" else args.effort_weight
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for state, reference, previous, target in train_loader:
            state = state.to(device)
            reference = reference.to(device)
            previous = previous.to(device)
            target = target.to(device)
            wrench, _, normalized = model(state, reference, previous)
            wrench_error = (wrench - target) / model.wrench_scale
            wrench_loss = wrench_error.square().mean()
            effort_loss = normalized.square().mean()
            loss = wrench_loss + effort_weight * effort_loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(state)
            count += len(state)
        metrics = evaluate(model, val_loader, device)
        metrics.update(epoch=epoch, train_loss=total_loss / count)
        history.append(metrics)
        score = metrics["rmse_fx"] + metrics["rmse_fy"] + metrics["rmse_nz"]
        checkpoint = {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "contract": CONTRACT.to_dict(),
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(checkpoint, output / "last.pt")
        if score < best_score:
            best_score = score
            torch.save(checkpoint, output / "best.pt")
        print(json.dumps(metrics, sort_keys=True))

    with (output / "learning_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("full", "overfit"),
                        default="full")
    parser.add_argument("--overfit-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--effort-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if (args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or
            args.overfit_samples <= 0):
        parser.error("epochs, batch size, and learning rate must be positive")
    if args.mode == "full" and not args.val:
        parser.error("full mode requires one or more --val datasets")
    train(args)


if __name__ == "__main__":
    main()
