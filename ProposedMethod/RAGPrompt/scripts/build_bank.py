"""Build the memory bank (training split only) for one dataset.

    python scripts/build_bank.py --dataset sims
    python scripts/build_bank.py --dataset iemocap --audio_len 350
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmm.bank import MODALITIES, MemoryBank  # noqa: E402
from ragmm.config import DATASETS, bank_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--audio_len", type=int, default=350, help="IEMOCAP only")
    parser.add_argument("--n_segments", type=int, default=16, help="time segments per stored value")
    args = parser.parse_args()

    bank = MemoryBank.build(args.dataset, audio_len=args.audio_len, n_segments=args.n_segments)
    path = bank_path(args.dataset, args.audio_len)
    bank.save(path)

    print(f"Saved {path}")
    print(f"  samples: {len(bank)}  groups: {bank.meta['n_groups']}  size: {bank.size_mb():.1f} MB")
    for m in MODALITIES:
        print(f"  {m}: key {tuple(bank.keys[m].shape)}  value {tuple(bank.values[m].shape)}")


if __name__ == "__main__":
    main()
