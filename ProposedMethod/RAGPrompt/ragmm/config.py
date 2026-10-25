"""Paths and per-dataset settings shared by every script.

All paths are absolute, so scripts can be launched from any directory.
"""

from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]  # .../RAGPrompt
PROJECT_ROOT = RAG_ROOT.parent  # .../MLProjectVidit
MPLMM_DIR = PROJECT_ROOT / "MPLMM"

PRETRAINED_BACKBONE = MPLMM_DIR / "pretrained" / "mosei.pt"

BANK_DIR = RAG_ROOT / "banks"
CHECKPOINT_DIR = RAG_ROOT / "checkpoints"
RESULTS_DIR = RAG_ROOT / "results"
RUNS_CSV = RESULTS_DIR / "runs.csv"

# positive: how a regression score is turned into a binary label. These match
# the conventions of the final baseline table (results/table_main_sims_fixed.csv).
DATASETS = {
    "mosi": {
        "data_path": MPLMM_DIR / "dataset" / "mosi_data.pkl",
        "task": "regression",
        "output_dim": 1,
        "positive": ">=0",
    },
    "mosei": {
        "data_path": MPLMM_DIR / "dataset" / "mosei_senti_data.pkl",
        "task": "regression",
        "output_dim": 1,
        "positive": ">=0",
    },
    "sims": {
        "data_path": MPLMM_DIR / "dataset" / "sims_data.pkl",
        "task": "regression",
        "output_dim": 1,
        "positive": ">0",
    },
    "iemocap": {
        "data_path": MPLMM_DIR / "dataset" / "iemocap" / "IEMOCAP_features_2021",
        "task": "classification",
        "output_dim": 4,
        "positive": None,
    },
}

# Same defaults as MPLMM/main.py, so the backbone is built exactly as in the baseline.
MODEL_DEFAULTS = {
    "attn_dropout": 0.1,
    "attn_dropout_a": 0.1,
    "attn_dropout_v": 0.1,
    "relu_dropout": 0.1,
    "embed_dropout": 0.25,
    "res_dropout": 0.1,
    "out_dropout": 0.1,
    "layers": 5,
    "proj_dim": 30,
    "num_heads": 5,
    "attn_mask": True,
    "prompt_dim": 30,
    "prompt_length": 16,
}

# Table columns are the AVAILABLE modalities; values are MPLMM missing-mode codes.
CASES = {"{a}": 4, "{v}": 3, "{t}": 5, "{a,v}": 0, "{a,t}": 2, "{v,t}": 1}
FULL_MODE = 6


def dataset_cfg(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}', expected one of {list(DATASETS)}")
    return DATASETS[name]


def bank_path(dataset, audio_len=350):
    suffix = f"_a{audio_len}" if dataset == "iemocap" else ""
    return BANK_DIR / f"{dataset}{suffix}_bank.pt"
