"""Go / no-go check BEFORE any training: does retrieval carry real signal?

For every validation sample and each of the 6 missing-modality cases, retrieve
K training samples using only the AVAILABLE modalities, then measure

  kNN ACC     label predicted from the neighbours' labels (similarity-weighted)
              vs the true label; compare with the majority-class baseline
  miss. cos   cosine between the TRUE missing-modality key and the weighted
              mean of the RETRIEVED missing-modality keys (how well retrieval
              "guesses" the missing modality)

each for real retrieval and for random retrieval (mean of several draws).
"strict" (only when the split shares videos with training, i.e. CH-SIMS) is
real retrieval that may not use clips of the query's own video.
If real retrieval does not beat random, retrieval adds nothing and RAG is not
worth training.

    python scripts/check_retrieval.py --dataset sims
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ragmm.bank import MODE_SPEC, MemoryBank  # noqa: E402
from ragmm.config import CASES, DATASETS, RESULTS_DIR, bank_path, dataset_cfg  # noqa: E402
from ragmm.data import group_names, load_split, make_loader, to_device  # noqa: E402
from ragmm.evaluate import to_labels  # noqa: E402


@torch.no_grad()
def split_queries(bank, dataset, split, audio_len, device):
    data = load_split(dataset, split, full_data=True, audio_len=audio_len)
    queries, labels = {m: [] for m in "lav"}, []
    for batch in make_loader(data, batch_size=128):
        (text, audio, vision), y, _ = to_device(batch, device)
        for m, q in bank.query_keys(text, audio, vision).items():
            queries[m].append(q)
        labels.append(y.reshape(-1))
    return {m: torch.cat(q) for m, q in queries.items()}, torch.cat(labels)


def knn_predict(bank, indices, log_w, task, n_classes):
    w = log_w.exp()
    neighbours = bank.labels[indices]
    if task == "classification":
        votes = torch.zeros(indices.shape[0], n_classes, device=w.device)
        votes.scatter_add_(1, neighbours.long(), w)
        return votes.argmax(1)
    return (w * neighbours.float()).sum(1)


def missing_cosine(bank, queries, indices, log_w, missing):
    sims = []
    for m in missing:
        guess = (log_w.exp().unsqueeze(-1) * bank.keys[m][indices]).sum(1)
        sims.append(F.cosine_similarity(guess, queries[m], dim=1).mean().item())
    return float(np.mean(sims))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--split", default="valid", choices=["valid", "test"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--audio_len", type=int, default=350)
    parser.add_argument("--random_draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    cfg = dataset_cfg(args.dataset)
    bank = MemoryBank.load(bank_path(args.dataset, args.audio_len), device)
    queries, labels = split_queries(bank, args.dataset, args.split, args.audio_len, device)

    y_true = to_labels(args.dataset, labels.cpu().numpy())
    majority = np.bincount(y_true.astype(int)).max() / len(y_true) * 100

    split_groups = bank.group_ids(group_names(args.dataset, args.split))
    conditions = [("real", False, 1, None), ("random", True, args.random_draws, None)]
    if (split_groups >= 0).any():
        conditions.append(("strict", False, 1, split_groups))

    rows = []
    for case, mode in CASES.items():
        available, missing = MODE_SPEC[mode]
        row = {"case": case}
        for name, random, draws, excl in conditions:
            accs, coss = [], []
            for _ in range(draws):
                idx, log_w = bank.retrieve(queries, available, args.k, args.tau,
                                           exclude_groups=excl, random=random)
                pred = knn_predict(bank, idx, log_w, cfg["task"], cfg["output_dim"])
                y_pred = to_labels(args.dataset, pred.cpu().numpy())
                accs.append((y_pred == y_true).mean() * 100)
                coss.append(missing_cosine(bank, queries, idx, log_w, missing))
            row[f"kNN ACC {name}"] = round(float(np.mean(accs)), 2)
            row[f"miss. cos {name}"] = round(float(np.mean(coss)), 3)
        rows.append(row)

    print(f"{args.dataset} [{args.split}]  K={args.k}  bank={len(bank)}  majority-class ACC={majority:.2f}")
    names = [c[0] for c in conditions]
    print(f"{'case':7s}" + "".join(f"{'kNN ACC ' + n:>17s}" for n in names)
          + "".join(f"{'miss.cos ' + n:>17s}" for n in names))
    for r in rows:
        print(f"{r['case']:7s}" + "".join(f"{r['kNN ACC ' + n]:17.2f}" for n in names)
              + "".join(f"{r['miss. cos ' + n]:17.3f}" for n in names))

    out = RESULTS_DIR / f"retrieval_check_{args.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
