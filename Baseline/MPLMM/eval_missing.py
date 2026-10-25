"""Per-case missing-modality evaluation for MOSI checkpoints.

Same protocol as eval_mosi() in BaselineReproduction.ipynb: every test sample is
evaluated once per fixed missing-modality case, binary accuracy/F1 with >= 0 as
positive, and the six cases are averaged.

Example:
    python eval_missing.py --checkpoint results/mosi_spatial_seed_1111.pt \
        --data_path C:/mplmm_data/mosi_data.pkl --spatial_feats C:/mplmm_data/mosi_spatial_t16
"""
import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.utils import get_data

# Table columns are the AVAILABLE modalities (see PromptModel.get_complete_data).
CASES = {"{a}": 4, "{v}": 3, "{t}": 5, "{a,v}": 0, "{a,t}": 2, "{v,t}": 1}


def evaluate(model, loader, device):
    row = {}
    model.eval()
    for case, mode in CASES.items():
        preds, truths = [], []
        with torch.no_grad():
            for (text, audio, vision), y, _ in loader:
                modes = torch.full((text.size(0),), mode, dtype=torch.long, device=device)
                out = model(text.to(device), audio.to(device), vision.to(device), modes)
                preds.append(out.cpu().view(-1))
                truths.append(y.cpu().view(-1))
        p, t = torch.cat(preds).numpy() >= 0, torch.cat(truths).numpy() >= 0
        row[f"{case} ACC"] = accuracy_score(t, p) * 100
        row[f"{case} F1"] = f1_score(t, p, average="weighted", zero_division=0) * 100
    row["Avg. ACC"] = np.mean([row[f"{c} ACC"] for c in CASES])
    row["Avg. F1"] = np.mean([row[f"{c} F1"] for c in CASES])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--spatial_feats", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--label", default=None, help="row name in the printed table / csv")
    parser.add_argument("--out_csv", default=None, help="append the row to this csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False).to(device)
    data_args = SimpleNamespace(dataset="mosi", data_path=args.data_path, drop_rate=0.0,
                                spatial_feats=args.spatial_feats)
    loader = DataLoader(get_data(data_args, split=args.split, full_data=True), batch_size=64, shuffle=False)
    row = evaluate(model, loader, device)

    label = args.label or os.path.splitext(os.path.basename(args.checkpoint))[0]
    print(f"\n{label} ({args.split})")
    print("-" * 50)
    for case in CASES:
        print(f"{case:>6} | ACC: {row[f'{case} ACC']:6.2f} | F1: {row[f'{case} F1']:6.2f}")
    print(f"{'Avg.':>6} | ACC: {row['Avg. ACC']:6.2f} | F1: {row['Avg. F1']:6.2f}")

    if args.out_csv:
        new_file = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Split", *row])
            if new_file:
                writer.writeheader()
            writer.writerow({"Model": label, "Split": args.split, **{k: round(v, 2) for k, v in row.items()}})


if __name__ == "__main__":
    main()
