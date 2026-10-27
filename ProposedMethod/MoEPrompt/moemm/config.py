"""Paths and per-dataset settings shared by every script.

All paths are absolute, so scripts can be launched from any directory.
"""

from pathlib import Path

MOE_ROOT = Path(__file__).resolve().parents[1]  # .../MoEPrompt
PROJECT_ROOT = MOE_ROOT.parent  # .../MLProjectVidit
MPLMM_DIR = PROJECT_ROOT / "MPLMM"

PRETRAINED_BACKBONE = MPLMM_DIR / "pretrained" / "mosei.pt"

CHECKPOINT_DIR = MOE_ROOT / "checkpoints"
RESULTS_DIR = MOE_ROOT / "results"
RUNS_CSV = RESULTS_DIR / "runs.csv"

# positive: how a regression score is turned into a binary label. These match
# the conventions of the final baseline table (MPLMM/results/table_main_sims_fixed.csv).
# legacy_ckpt: the baseline checkpoint behind that table (used by scripts/diagnose.py).
DATASETS = {
    "mosi": {
        "data_path": MPLMM_DIR / "dataset" / "mosi_data.pkl",
        "task": "regression",
        "output_dim": 1,
        "positive": ">=0",
        "legacy_ckpt": MPLMM_DIR / "results" / "mosi_stage2_shuffle_30ep_seed_1111.pt",
    },
    "mosei": {
        "data_path": MPLMM_DIR / "dataset" / "mosei_senti_data.pkl",
        "task": "regression",
        "output_dim": 1,
        "positive": ">=0",
        "legacy_ckpt": MPLMM_DIR / "results" / "mosei_stage2_seed_1111_30ep.pt",
    },
    "sims": {
        "data_path": MPLMM_DIR / "dataset" / "sims_data.pkl",
        "task": "regression",
        "output_dim": 1,
        "positive": ">0",
        "legacy_ckpt": MPLMM_DIR / "results" / "sims_fixed" / "stage2_lr0.001_seed1111.pt",
    },
    "iemocap": {
        "data_path": MPLMM_DIR / "dataset" / "iemocap" / "IEMOCAP_features_2021",
        "task": "classification",
        "output_dim": 4,
        "positive": None,
        # The table's IEMOCAP row was trained and evaluated with 150 audio frames.
        "legacy_ckpt": MPLMM_DIR / "results" / "iemocap_stage2_shuffle_a150_seed_1111_30ep.pt",
        "audio_len": 150,
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


def audio_len(name, override=None):
    """IEMOCAP audio frames: the explicit value, else the dataset's setting (only
    IEMOCAP's loader uses it; the other datasets ignore it)."""
    return override if override is not None else DATASETS[name].get("audio_len", 350)
