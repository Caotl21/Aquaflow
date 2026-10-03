"""Train hofa_policy with DAgger sequences and multi-step action rollout."""
import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .actuation import ThrusterGeometry
from .contract import CONTRACT
from .dagger_dataset import DaggerSequenceDataset
from .model import HofaPolicyNet, ModelConfig


def rollout_loss(model, batch, teacher_probability, axis_weights):
    state, reference, previous, target = batch
    prediction = None
    losses = []
    for step in range(state.shape[1]):
        if step == 0:
            current_previous = previous
        else:
            use_teacher = (
                torch.rand(state.shape[0], device=state.device)
                < teacher_probability)
            current_previous = torch.where(
                use_teacher.unsqueeze(1), target[:, step - 1],
                prediction.detach())
        prediction, _, _ = model(
            state[:, step], reference[:, step], current_previous)
        error = (prediction - target[:, step]) / model.wrench_scale
        losses.append((error.square() * axis_weights).mean())
    return torch.stack(losses).mean(), prediction


def evaluate(model, loader, device, axis_weights):
    model.eval()
    total = []
    with torch.inference_mode():
        for batch in loader:
            batch = tuple(value.to(device) for value in batch)
            prediction = None
            errors = []
            for step in range(batch[0].shape[1]):
                previous = (
                    batch[2] if step == 0 else prediction)
                prediction, _, _ = model(
                    batch[0][:, step], batch[1][:, step], previous)
                errors.append(prediction - batch[3][:, step])
            total.append(torch.stack(errors, dim=1).cpu())
    errors = torch.cat(total)
    return {
        "rmse_fx": float(errors[:, :, 0].square().mean().sqrt()),
        "rmse_fy": float(errors[:, :, 1].square().mean().sqrt()),
        "rmse_nz": float(errors[:, :, 2].square().mean().sqrt()),
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
    train_data = DaggerSequenceDataset(args.train, args.rollout_steps)
    val_data = DaggerSequenceDataset(args.val, args.rollout_steps)
    if not len(train_data) or not len(val_data):
        raise ValueError("DAgger train/val sequences are empty")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=generator)
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    init_checkpoint = None
    checkpoint = None
    if args.init_checkpoint:
        init_checkpoint = Path(args.init_checkpoint).resolve()
        if not init_checkpoint.is_file():
            raise FileNotFoundError(init_checkpoint)
        checkpoint = torch.load(
            init_checkpoint, map_location=device, weights_only=True)
        if checkpoint.get("contract") != CONTRACT.to_dict():
            raise ValueError("initial checkpoint contract mismatch")
        config = ModelConfig(**checkpoint["model_config"])
    else:
        config = ModelConfig()
    model = HofaPolicyNet(
        ThrusterGeometry.bricsbot(), config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    axis_weights = torch.tensor(
        [args.fx_weight, args.fy_weight, args.nz_weight],
        dtype=torch.float32, device=device)
    run = {
        "schema": "hofa_policy_dagger_training_v1",
        "contract": CONTRACT.to_dict(),
        "rollout_steps": args.rollout_steps,
        "teacher_probability_start": args.teacher_probability_start,
        "teacher_probability_end": args.teacher_probability_end,
        "axis_weights": axis_weights.tolist(),
        "train_sequences": len(train_data),
        "val_sequences": len(val_data),
        "train_sources": train_data.sources,
        "val_sources": val_data.sources,
        "seed": args.seed,
        "init_checkpoint": (
            str(init_checkpoint) if init_checkpoint is not None else None),
        "init_epoch": (
            int(checkpoint.get("epoch", -1))
            if checkpoint is not None else None),
        "model_config": config.to_dict(),
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n")
    history = []
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        fraction = (epoch - 1) / max(1, args.epochs - 1)
        teacher_probability = (
            args.teacher_probability_start * (1.0 - fraction) +
            args.teacher_probability_end * fraction)
        total, count = 0.0, 0
        for batch in train_loader:
            batch = tuple(value.to(device) for value in batch)
            loss, _ = rollout_loss(
                model, batch, teacher_probability, axis_weights)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite DAgger loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch[0])
            count += len(batch[0])
        metrics = evaluate(model, val_loader, device, axis_weights)
        metrics.update(
            epoch=epoch, train_loss=total / count,
            teacher_probability=teacher_probability)
        history.append(metrics)
        score = metrics["rmse_fx"] + metrics["rmse_fy"] + metrics["rmse_nz"]
        checkpoint = {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "contract": CONTRACT.to_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "training": run,
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
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--init-checkpoint",
        help="initialize from an existing BC checkpoint before fine-tuning")
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--teacher-probability-start", type=float, default=0.8)
    parser.add_argument("--teacher-probability-end", type=float, default=0.2)
    parser.add_argument("--fx-weight", type=float, default=1.5)
    parser.add_argument("--fy-weight", type=float, default=1.0)
    parser.add_argument("--nz-weight", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.rollout_steps <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error("rollout steps, epochs, and batch size must be positive")
    if not (0.0 <= args.teacher_probability_end <=
            args.teacher_probability_start <= 1.0):
        parser.error("teacher probabilities must satisfy 0 <= end <= start <= 1")
    train(args)


if __name__ == "__main__":
    main()
