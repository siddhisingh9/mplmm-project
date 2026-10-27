"""Stage 0 (no training): how much room is there for a better generator?

Re-evaluates the baseline checkpoint behind the final table three times:

  baseline : unchanged (must reproduce the table, otherwise stop)
  oracle   : generated features replaced by the projected REAL modality.
             No generator can do better than feeding the true features, so
             oracle - baseline = the headroom for MoE / reconstruction.
  zero     : generated features replaced by zeros. baseline - zero = how much
             the current generator contributes at all.

Only the generator output changes; prompts, backbone and head are untouched.
Note: the checkpoint was trained with its own generator's outputs, so the
oracle is a lower estimate of the real headroom.

    python scripts/diagnose.py --dataset mosi
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from moemm.config import CASES, DATASETS, RESULTS_DIR, audio_len, dataset_cfg  # noqa: E402
from moemm.data import load_split  # noqa: E402
from moemm.evaluate import evaluate_cases, flatten, format_results  # noqa: E402
from moemm.model import from_legacy, make_hyp_params  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--ckpt", type=Path, help="baseline checkpoint (default: the table's)")
    parser.add_argument("--audio_len", type=int, default=None,
                        help="IEMOCAP audio frames (default: dataset setting, 150)")
    args = parser.parse_args()
    args.audio_len = audio_len(args.dataset, args.audio_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or dataset_cfg(args.dataset)["legacy_ckpt"]
    train = load_split(args.dataset, "train", audio_len=args.audio_len)
    model = from_legacy(ckpt, make_hyp_params(args.dataset, train), device)
    print(f"{args.dataset} [{args.split}]  checkpoint {ckpt}")

    rows = []
    for override in (None, "oracle", "zero"):
        model.generator_override = override
        name = override or "baseline"
        res = evaluate_cases(model, args.dataset, args.split, device, audio_len=args.audio_len)
        print(f"\n{name}\n{format_results(res)}")
        rows.append({"dataset": args.dataset, "split": args.split, "variant": name, **flatten(res)})

    base, oracle, zero = rows
    print("\nAvg. ACC / F1 differences")
    for label, a, b in (("oracle - baseline (headroom)", oracle, base),
                        ("baseline - zero (generator use)", base, zero)):
        print(f"  {label:32s} {a['Avg. ACC'] - b['Avg. ACC']:+6.2f} / {a['Avg. F1'] - b['Avg. F1']:+6.2f}")
    print("Per case, oracle - baseline ACC: "
          + ", ".join(f"{c} {oracle[f'{c} ACC'] - base[f'{c} ACC']:+.2f}" for c in CASES))

    out = RESULTS_DIR / f"diagnose_{args.dataset}_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
