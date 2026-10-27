"""Data loading through MPLMM's own dataset classes.

Features come from MPLMM's loaders, so preprocessing (e.g. CH-SIMS
normalisation, IEMOCAP audio padding) is identical to the baseline.

Note: a "missing" modality is only a mode code. The tensors of every modality
are always present in the batch, so model code must never read a modality the
mode marks as missing (scripts/sanity_checks.py tests this). Only the
reconstruction loss may use it, as a training target.
"""

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from moemm.bridge import get_data
from moemm.config import dataset_cfg


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


def make_loader(dataset_obj, batch_size=64, shuffle=False, seed=1111):
    generator = None
    if shuffle:
        # MPLMM sets the default device to CUDA; the sampler's generator must match it.
        generator = torch.Generator(device=torch.get_default_device())
        generator.manual_seed(seed)
    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=getattr(dataset_obj, "collate_fn", None),  # IEMOCAP pads sequences
    )


def to_device(batch, device):
    """(text, audio, vision), y, mode -> same structure on `device`."""
    (text, audio, vision), y, mode = batch
    return (
        (text.to(device).float(), audio.to(device).float(), vision.to(device).float()),
        y.to(device),
        torch.as_tensor(mode).to(device).long(),
    )
