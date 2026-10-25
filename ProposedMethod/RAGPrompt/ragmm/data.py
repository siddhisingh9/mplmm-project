"""Data loading on top of MPLMM's own dataset classes.

Features come from MPLMM's loaders, so preprocessing (e.g. CH-SIMS
normalisation, IEMOCAP audio padding) is identical to the baseline. This module
only adds:
  * group ids (video / dialogue) used to stop a training sample retrieving
    itself or clips from its own video,
  * an index wrapper so the training loop knows which bank entry each sample is.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from ragmm.bridge import get_data
from ragmm.config import dataset_cfg

IEMOCAP_SPLITS = {"train": "trn", "valid": "val", "test": "tst"}


def mplmm_args(dataset, drop_rate=0.0, audio_len=350, seed=1111):
    """The argument namespace MPLMM's get_data() expects."""
    return SimpleNamespace(
        dataset=dataset,
        data_path=str(dataset_cfg(dataset)["data_path"]),
        drop_rate=drop_rate,
        sims_norm=1,
        iemo_audio_len=audio_len,
        seed=seed,
    )


def load_split(dataset, split, drop_rate=0.0, full_data=False, audio_len=350):
    """full_data=True gives every sample missing mode 6 (all modalities present)."""
    args = mplmm_args(dataset, drop_rate=drop_rate, audio_len=audio_len)
    return get_data(args, split=split, full_data=full_data)


def group_names(dataset, split):
    """One group name per sample, in dataset order: the source video or dialogue."""
    path = dataset_cfg(dataset)["data_path"]
    if dataset == "iemocap":
        names = np.load(path / "target" / "1" / f"{IEMOCAP_SPLITS[split]}_int2name.npy")
        utterances = [n[0].decode() for n in names]  # e.g. Ses02F_script03_1_M001
        return [u.rsplit("_", 1)[0] for u in utterances]  # dialogue Ses02F_script03_1

    with open(path, "rb") as f:
        ids = pickle.load(f)[split]["id"]
    if dataset == "sims":  # 'video_0018$_$0004'
        return [str(i).split("$_$")[0] for i in ids]
    if dataset == "mosei":  # ['-3g5yACwYnA', start, end]
        return [str(row[0]) for row in ids]
    # mosi: [b'03bSnISJMiM_11', start, end] -> video '03bSnISJMiM'
    return [row[0].decode().rsplit("_", 1)[0] for row in ids]


class IndexedDataset(Dataset):
    """Wraps an MPLMM dataset so each item also carries its index."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, mode = self.base[idx]
        return x, y, mode, idx

    def collate(self, batch):
        items = [b[:3] for b in batch]
        base_collate = getattr(self.base, "collate_fn", None)  # IEMOCAP pads sequences
        x, y, mode = base_collate(items) if base_collate else default_collate(items)
        idx = torch.tensor([b[3] for b in batch], dtype=torch.long)
        return x, y, mode, idx


def make_loader(dataset_obj, batch_size=64, shuffle=False, seed=1111, with_index=False):
    data = IndexedDataset(dataset_obj) if with_index else dataset_obj
    if with_index:
        collate = data.collate
    else:
        collate = getattr(dataset_obj, "collate_fn", None)

    generator = None
    if shuffle:
        # MPLMM sets the default device to CUDA; the sampler's generator must match it.
        generator = torch.Generator(device=torch.get_default_device())
        generator.manual_seed(seed)
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, generator=generator, collate_fn=collate
    )


def to_device(batch, device):
    """(text, audio, vision), y, mode[, idx] -> same structure on `device`."""
    (text, audio, vision), y, mode, *rest = batch
    out = [
        (text.to(device).float(), audio.to(device).float(), vision.to(device).float()),
        y.to(device),
        torch.as_tensor(mode).to(device).long(),
    ]
    return out + [r.to(device) for r in rest]
