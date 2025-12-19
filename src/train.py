import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from data_loader import load_kaggle_cardio_bundle, CardioDataset
from model import MLPBinaryClassifier
from utils import set_seed, get_device, ensure_dir


def compute_metrics_from_logits(logits: torch.Tensor, y_true: torch.Tensor):
    """
    Returns loss-friendly outputs + numpy probabilities for sklearn-like metrics if needed later.
    For now we log: accuracy + avg prob.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == y_true).float().mean().item()
    avg_prob = probs.mean().item()
    return acc, avg_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    device = get_device(cfg["train"]["device"])
    print(f"✅ Using device: {device}")

    # W&B init
    use_wandb = bool(cfg.get("wandb", {}).get("enabled", True))
    if use_wandb:
        wandb.init(
            project=cfg["project"]["name"],
            name=cfg["project"].get("run_name"),
            config=cfg,
        )

    # Load data
    bundle = load_kaggle_cardio_bundle(
        csv_path=cfg["data"]["raw_csv_path"],
        sep=cfg["data"].get("sep", ";"),
        target_col=cfg["data"]["target_col"],
        id_col=cfg["data"].get("id_col"),
        test_size=float(cfg["split"]["test_size"]),
        seed=seed,
        stratify=bool(cfg["split"].get("stratify", True)),
    )

    # Dataset stats
    pos_rate = float(np.mean(bundle.y_train))
    if use_wandb:
        wandb.log({
            "data/n_train": int(bundle.X_train.shape[0]),
            "data/n_test": int(bundle.X_test.shape[0]),
            "data/n_features": int(bundle.X_train.shape[1]),
            "data/pos_rate_train": pos_rate,
        })

    train_ds = CardioDataset(bundle.X_train, bundle.y_train)
    test_ds = CardioDataset(bundle.X_test, bundle.y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = MLPBinaryClassifier(
        input_dim=bundle.X_train.shape[1],
        hidden_dims=tuple(cfg["model"].get("hidden_dims", [128, 64])),
        dropout=float(cfg["model"].get("dropout", 0.2)),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    ensure_dir("models")

    log_every = int(cfg.get("wandb", {}).get("log_every_steps", 20))
    global_step = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if use_wandb and (global_step % log_every == 0):
                acc, avg_prob = compute_metrics_from_logits(logits.detach(), yb.detach())
                wandb.log({
                    "train/loss_step": float(loss.item()),
                    "train/acc_step": float(acc),
                    "train/avg_prob_step": float(avg_prob),
                    "global_step": global_step,
                })

        # Eval
        model.eval()
        test_losses, test_accs = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                acc, _ = compute_metrics_from_logits(logits, yb)
                test_losses.append(loss.item())
                test_accs.append(acc)

        train_loss_epoch = running_loss / max(1, len(train_loader))
        test_loss_epoch = float(np.mean(test_losses))
        test_acc_epoch = float(np.mean(test_accs))

        print(f"Epoch {epoch:02d} | train_loss={train_loss_epoch:.4f} | test_loss={test_loss_epoch:.4f} | test_acc={test_acc_epoch:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss_epoch": float(train_loss_epoch),
                "test/loss_epoch": float(test_loss_epoch),
                "test/acc_epoch": float(test_acc_epoch),
            })

        # Save checkpoint each epoch (simple)
        ckpt_path = f"models/mlp_epoch_{epoch:02d}.pt"
        torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
        if use_wandb:
            wandb.save(ckpt_path)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
