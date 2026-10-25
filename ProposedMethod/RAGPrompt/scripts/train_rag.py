"""Train one configuration and record its validation and test results.

  --mode static   generative prompt is the baseline's static prompt (RAG off)
  --mode rag      prompt built from the top-K retrieved training samples
  --mode random   control: K RANDOM training samples (same extra parameters)

All three use the same frozen MOSEI backbone and training loop as the baseline
(MPLMM/src/train.py): Adam, L1 / cross-entropy loss, grad clip 0.8,
ReduceLROnPlateau, best checkpoint chosen on validation loss. The test split is
only used once, after training. One row is appended to results/runs.csv.

    python scripts/train_rag.py --dataset sims --mode rag --k 5 --seed 1111
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

from ragmm.bank import MemoryBank  # noqa: E402
from ragmm.config import CHECKPOINT_DIR, DATASETS, RUNS_CSV, bank_path, dataset_cfg  # noqa: E402
from ragmm.data import load_split, make_loader, to_device  # noqa: E402
from ragmm.evaluate import evaluate_cases, flatten, format_results, shares_groups_with_bank  # noqa: E402
from ragmm.model import build_model, load_checkpoint, save_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--mode", required=True, choices=["static", "rag", "random"])
    p.add_argument("--k", type=int, default=5, help="neighbours retrieved per sample")
    p.add_argument("--tau", type=float, default=0.1, help="softmax temperature on similarities")
    p.add_argument("--exclude", default="group", choices=["group", "self"],
                   help="training: never retrieve the own video/dialogue (group) or only "
                        "the sample itself (self)")
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--drop_rate", type=float, default=0.7, help="training missing rate")
    p.add_argument("--shuffle", type=int, default=1, help="shuffle the training split")
    p.add_argument("--audio_len", type=int, default=350, help="IEMOCAP audio frames")
    p.add_argument("--clip", type=float, default=0.8)
    p.add_argument("--when", type=int, default=10, help="LR scheduler patience")
    p.add_argument("--log_every", type=int, default=0, help="print train loss every N batches (0: off)")
    return p.parse_args()


def run_name(args):
    if args.mode == "static":
        return f"{args.dataset}_static_seed{args.seed}"
    excl = "" if args.exclude == "group" else "_exself"
    return f"{args.dataset}_{args.mode}_k{args.k}{excl}_seed{args.seed}"


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def loss_inputs(preds, y, cfg):
    if cfg["task"] == "classification":
        return preds.view(-1, cfg["output_dim"]), y.view(-1).long()
    target = y.squeeze(-1)  # [B, 1, 1] -> [B, 1]
    if preds.shape != target.shape:
        raise RuntimeError(f"preds {tuple(preds.shape)} vs labels {tuple(target.shape)}")
    return preds, target


def run_epoch(model, loader, cfg, criterion, device, bank=None, optimizer=None, clip=None,
              log_every=0, exclude="group"):
    """One pass. With an optimizer: training (indexed loader); without: validation."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for step, batch in enumerate(loader):
        if training:
            (text, audio, vision), y, modes, idx = to_device(batch, device)
            excl = {}
            if bank is not None:  # a training sample is in the bank: never retrieve it
                excl = ({"exclude_groups": bank.groups[idx]} if exclude == "group"
                        else {"exclude_indices": idx})
        else:
            (text, audio, vision), y, modes = to_device(batch, device)
            excl = {}  # validation samples are not in the bank

        with torch.set_grad_enabled(training):
            preds = model(text, audio, vision, modes, **excl)
            loss = criterion(*loss_inputs(preds, y, cfg))
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if log_every and step % log_every == 0:
                print(f"    batch {step:4d}/{len(loader)}  loss {loss.item():.4f}")
        total += loss.item() * text.shape[0]
        count += text.shape[0]
    return total / count


def append_run(row):
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    cfg = dataset_cfg(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)
    name = run_name(args)
    ckpt_path = CHECKPOINT_DIR / f"{name}.pt"
    print(f"=== {name} ===")

    train = load_split(args.dataset, "train", drop_rate=args.drop_rate, audio_len=args.audio_len)
    valid = load_split(args.dataset, "valid", drop_rate=args.drop_rate, audio_len=args.audio_len)
    train_loader = make_loader(train, args.batch_size, shuffle=bool(args.shuffle), seed=args.seed,
                               with_index=True)
    valid_loader = make_loader(valid, args.batch_size)

    rag_cfg = {"enabled": args.mode != "static", "random": args.mode == "random",
               "k": args.k, "tau": args.tau}
    model, hyp_params = build_model(args.dataset, train, rag_cfg, device)
    bank = None
    if rag_cfg["enabled"]:
        bank = MemoryBank.load(bank_path(args.dataset, args.audio_len), device)
        if len(bank) != len(train):
            raise RuntimeError(f"bank has {len(bank)} entries, training split {len(train)}")
        model.attach_bank(bank)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}  (new rag_*: "
          f"{sum(p.numel() for p in model.rag_parameters()) if rag_cfg['enabled'] else 0:,})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.when, factor=0.1)
    criterion = nn.CrossEntropyLoss() if cfg["task"] == "classification" else nn.L1Loss()

    best_val, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, cfg, criterion, device, bank, optimizer,
                               args.clip, args.log_every, args.exclude)
        val_loss = run_epoch(model, valid_loader, cfg, criterion, device)
        scheduler.step(val_loss)
        saved = val_loss < best_val
        if saved:
            best_val, best_epoch = val_loss, epoch
            save_checkpoint(model, ckpt_path, hyp_params, {"args": vars(args), "epoch": epoch})
        gates = (torch.tanh(model.rag_gate).tolist() if rag_cfg["enabled"] else None)
        print(f"epoch {epoch:3d} | {time.time() - start:5.1f}s | train {train_loss:.4f} | "
              f"valid {val_loss:.4f}{' *' if saved else ''}"
              + (f" | gates {[round(g, 3) for g in gates]}" if gates else ""))

    model, _ = load_checkpoint(ckpt_path, device, bank)
    valid_res = evaluate_cases(model, args.dataset, "valid", device, audio_len=args.audio_len)
    test_res = evaluate_cases(model, args.dataset, "test", device, audio_len=args.audio_len)
    print(f"\nbest epoch {best_epoch} (valid loss {best_val:.4f})")
    print("VALID\n" + format_results(valid_res))
    print("TEST\n" + format_results(test_res))

    # Strict test: also no same-video neighbours at test time (only differs for CH-SIMS).
    strict = {f"strict {k}": "" for k in flatten(test_res, "test ")}
    if bank is not None and shares_groups_with_bank(bank, args.dataset, "test"):
        strict_res = evaluate_cases(model, args.dataset, "test", device, audio_len=args.audio_len,
                                    strict=True, bank=bank)
        print("TEST (strict: no same-video neighbours)\n" + format_results(strict_res))
        strict = flatten(strict_res, "strict test ")

    append_run({
        "run": name, "dataset": args.dataset, "mode": args.mode,
        "k": args.k if args.mode != "static" else "", "tau": args.tau if args.mode != "static" else "",
        "exclude": args.exclude if args.mode != "static" else "",
        "seed": args.seed, "epochs": args.epochs, "lr": args.lr, "best_epoch": best_epoch,
        "best_valid_loss": round(best_val, 4), "trainable_params": trainable,
        **flatten(valid_res, "valid "), **flatten(test_res, "test "), **strict,
    })
    print(f"Saved checkpoint {ckpt_path}\nAppended results to {RUNS_CSV}")


if __name__ == "__main__":
    main()
