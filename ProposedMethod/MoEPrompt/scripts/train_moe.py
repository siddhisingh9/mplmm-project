"""Train one configuration and record its validation and test results.

  --moe 0 --lambda_rec 0 --lambda_kd 0   baseline generator, task loss only (control)
  --moe 0 --lambda_rec R --lambda_kd D   Stage 1: H1 only (extra losses, no experts)
  --moe 1 --lambda_rec 0 --lambda_kd 0   Stage 2: H2 only (experts, task loss)
  --moe 1 --lambda_rec R --lambda_kd D   Stage 3: H1 + H2

    loss = task + lambda_rec * reconstruction + lambda_kd * distillation
                + lambda_bal * load balance (only with --moe 1)

Everything else is the baseline loop (MPLMM/src/train.py): frozen MOSEI
backbone, Adam, L1 / cross-entropy, grad clip 0.8, ReduceLROnPlateau, best
checkpoint on VALIDATION TASK LOSS (the extra losses never select the
checkpoint). The test split is only evaluated once, after training.
One row is appended to results/runs.csv; with --moe 1 the router weights per
case are also saved to results/routing/<run>.csv.

    python scripts/train_moe.py --dataset mosi --moe 1 --lambda_rec 0.5 --lambda_kd 0.5
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402

from moemm.config import (  # noqa: E402
    CHECKPOINT_DIR, DATASETS, FULL_MODE, RESULTS_DIR, RUNS_CSV, audio_len, dataset_cfg,
)
from moemm.data import load_split, make_loader, to_device  # noqa: E402
from moemm.evaluate import evaluate_cases, flatten, format_results, routing_table  # noqa: E402
from moemm.losses import distillation_loss, load_balance_loss, reconstruction_loss  # noqa: E402
from moemm.model import build_model, load_checkpoint, save_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--moe", type=int, default=1, choices=[0, 1], help="mixture of experts on/off")
    p.add_argument("--k", type=int, default=3, help="number of experts")
    p.add_argument("--rank", type=int, default=16, help="expert translator rank")
    p.add_argument("--tau", type=float, default=1.0, help="router softmax temperature")
    p.add_argument("--lambda_rec", type=float, default=0.0, help="reconstruction loss weight")
    p.add_argument("--lambda_kd", type=float, default=0.0, help="distillation loss weight")
    p.add_argument("--lambda_bal", type=float, default=0.01, help="load-balance weight (MoE only)")
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--drop_rate", type=float, default=0.7, help="training missing rate")
    p.add_argument("--shuffle", type=int, default=1, help="shuffle the training split")
    p.add_argument("--audio_len", type=int, default=None,
                   help="IEMOCAP audio frames (default: dataset setting, 150)")
    p.add_argument("--clip", type=float, default=0.8)
    p.add_argument("--when", type=int, default=10, help="LR scheduler patience")
    p.add_argument("--log_every", type=int, default=0, help="print losses every N batches (0: off)")
    p.add_argument("--max_batches", type=int, default=0, help="smoke test: stop epochs early (0: off)")
    args = p.parse_args()
    args.audio_len = audio_len(args.dataset, args.audio_len)
    return args


def run_name(args):
    core = f"moe_k{args.k}" if args.moe else "base"
    return f"{args.dataset}_{core}_rec{args.lambda_rec:g}_kd{args.lambda_kd:g}_seed{args.seed}"


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def task_loss(criterion, preds, y, cfg):
    if cfg["task"] == "classification":
        return criterion(preds.view(-1, cfg["output_dim"]), y.view(-1).long())
    target = y.squeeze(-1)  # [B, 1, 1] -> [B, 1]
    if preds.shape != target.shape:
        raise RuntimeError(f"preds {tuple(preds.shape)} vs labels {tuple(target.shape)}")
    return criterion(preds, target)


def train_epoch(model, loader, cfg, criterion, optimizer, args):
    """One training pass. Returns the mean of each loss term over the batches."""
    use_moe = model.moe_cfg["enabled"]
    sums, n_batches = {}, 0
    for step, batch in enumerate(loader):
        if args.max_batches and step >= args.max_batches:
            break
        (text, audio, vision), y, modes = to_device(batch, args.device)

        teacher = None
        if args.lambda_kd > 0 and (modes != FULL_MODE).any():
            # Teacher = the same model on the complete sample: eval mode (no dropout), no grad.
            model.eval()
            with torch.no_grad():
                _, teacher = model(text, audio, vision, torch.full_like(modes, FULL_MODE),
                                   return_aux=True)
        model.train()
        preds, aux = model(text, audio, vision, modes, return_aux=True)

        terms = {"task": task_loss(criterion, preds, y, cfg)}
        if args.lambda_rec > 0:
            terms["rec"] = reconstruction_loss(aux["recon"])
        if teacher is not None:
            terms["kd"] = distillation_loss(aux["features"], teacher["features"], modes)
        if use_moe and args.lambda_bal > 0:
            terms["bal"] = load_balance_loss(aux["probs"], modes)
        weights = {"task": 1.0, "rec": args.lambda_rec, "kd": args.lambda_kd, "bal": args.lambda_bal}
        terms = {k: v for k, v in terms.items() if v is not None}
        loss = sum(weights[k] * v for k, v in terms.items())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        for k, v in terms.items():
            sums[k] = sums.get(k, 0.0) + v.item()
        n_batches += 1
        if args.log_every and step % args.log_every == 0:
            print(f"    batch {step:4d}/{len(loader)}  "
                  + "  ".join(f"{k} {v.item():.4f}" for k, v in terms.items()))
    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def valid_loss(model, loader, cfg, criterion, device):
    """Task loss only, with MPLMM's random validation missing modes (as the baseline)."""
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        (text, audio, vision), y, modes = to_device(batch, device)
        loss = task_loss(criterion, model(text, audio, vision, modes), y, cfg)
        total += loss.item() * text.shape[0]
        count += text.shape[0]
    return total / count


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    cfg = dataset_cfg(args.dataset)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)
    name = run_name(args) + ("_smoke" if args.max_batches else "")
    ckpt_path = CHECKPOINT_DIR / f"{name}.pt"
    print(f"=== {name} ===")

    train = load_split(args.dataset, "train", drop_rate=args.drop_rate, audio_len=args.audio_len)
    valid = load_split(args.dataset, "valid", drop_rate=args.drop_rate, audio_len=args.audio_len)
    train_loader = make_loader(train, args.batch_size, shuffle=bool(args.shuffle), seed=args.seed)
    valid_loader = make_loader(valid, args.batch_size)

    moe_cfg = {"enabled": bool(args.moe), "k": args.k, "rank": args.rank, "tau": args.tau}
    model, hyp_params = build_model(args.dataset, train, moe_cfg, args.device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}  (new moe*: "
          f"{sum(p.numel() for p in model.moe_parameters()):,})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.when, factor=0.1)
    criterion = nn.CrossEntropyLoss() if cfg["task"] == "classification" else nn.L1Loss()

    best_val, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_terms = train_epoch(model, train_loader, cfg, criterion, optimizer, args)
        val = valid_loss(model, valid_loader, cfg, criterion, args.device)
        scheduler.step(val)
        saved = val < best_val
        if saved:
            best_val, best_epoch = val, epoch
            save_checkpoint(model, ckpt_path, hyp_params, {"args": vars(args) | {"device": str(args.device)},
                                                          "epoch": epoch})
        gates = f" | gates {[round(g, 3) for g in torch.tanh(model.moe_gate).tolist()]}" if args.moe else ""
        print(f"epoch {epoch:3d} | {time.time() - start:5.1f}s | "
              + " ".join(f"{k} {v:.4f}" for k, v in train_terms.items())
              + f" | valid {val:.4f}{' *' if saved else ''}{gates}")

    model, _ = load_checkpoint(ckpt_path, args.device)
    valid_res = evaluate_cases(model, args.dataset, "valid", args.device, audio_len=args.audio_len)
    test_res = evaluate_cases(model, args.dataset, "test", args.device, audio_len=args.audio_len)
    print(f"\nbest epoch {best_epoch} (valid loss {best_val:.4f})")
    print("VALID\n" + format_results(valid_res))
    print("TEST\n" + format_results(test_res))

    if args.moe:
        rows = routing_table(model, args.dataset, "test", args.device, audio_len=args.audio_len)
        print(f"gates (tanh) {[round(g, 3) for g in torch.tanh(model.moe_gate).tolist()]}")
        print("routing on test (mean weight per expert, entropy; uniform = "
              f"{np.log(args.k):.3f})")
        for r in rows:
            means = " ".join(f"{r[f'mean p{k + 1}']:.2f}" for k in range(args.k))
            print(f"  {r['case']:6s} gen {r['generated']}: p = [{means}]  entropy {r['entropy']:.3f}")
        out = RESULTS_DIR / "routing" / f"{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    append_csv(RESULTS_DIR / ("smoke_runs.csv" if args.max_batches else RUNS_CSV.name), {
        "run": name, "dataset": args.dataset, "moe": args.moe, "k": args.k if args.moe else "",
        "rank": args.rank if args.moe else "", "lambda_rec": args.lambda_rec,
        "lambda_kd": args.lambda_kd, "lambda_bal": args.lambda_bal if args.moe else "",
        "seed": args.seed, "epochs": args.epochs, "lr": args.lr, "best_epoch": best_epoch,
        "best_valid_loss": round(best_val, 4), "trainable_params": trainable,
        "gates": [round(g, 3) for g in torch.tanh(model.moe_gate).tolist()] if args.moe else "",
        **flatten(valid_res, "valid "), **flatten(test_res, "test "),
    })
    print(f"Saved checkpoint {ckpt_path}\nAppended results")


if __name__ == "__main__":
    main()
