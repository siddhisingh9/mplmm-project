"""Extra training losses (H1) and the router's load-balancing loss.

  total = task + lambda_rec * reconstruction + lambda_kd * distillation + lambda_bal * balance

Reconstruction and distillation use the normalised MSE
    NMSE = ||pred - target||^2 / ||target||^2
which is scale-free (1.0 means "as bad as predicting zeros"), so the lambdas
are comparable across datasets.

Reconstruction uses log(1 + NMSE) per (sample, modality) pair: same minimum
(0 at a perfect reconstruction), but bounded influence of badly scaled pairs.
At initialisation on MOSEI the generated text is ~7x larger than the real
projected text (NMSE ~77 for those pairs), which would otherwise dominate the
first updates.
"""

import torch

from moemm.config import FULL_MODE
from moemm.experts import AVAILABLE  # [7, 3] bool: modality present in each mode

EPS = 1e-6


def normalized_mse(pred, target):
    """Per-row normalised MSE, averaged over rows. Rows: first dimension."""
    dims = tuple(range(1, pred.dim()))
    err = (pred - target).pow(2).mean(dim=dims)
    scale = target.pow(2).mean(dim=dims).clamp(min=EPS)
    return (err / scale).mean()


def reconstruction_loss(recon):
    """recon: (generated, real) pairs, one per missing modality of each sample.

    Shapes differ between modalities, so every pair is normalised on its own.
    The real targets are already detached by the model.
    """
    if not recon:
        return None
    return torch.stack([torch.log1p(normalized_mse(gen, target)) for gen, target in recon]).mean()


def distillation_loss(student_features, teacher_features, modes):
    """Match the complete-data representation, only for samples with something missing."""
    sel = modes != FULL_MODE
    if not sel.any():
        return None
    return normalized_mse(student_features[sel], teacher_features[sel].detach())


def load_balance_loss(probs, modes):
    """KL(batch-mean routing || uniform) per target modality, over the samples where
    that modality is missing (the only samples whose routing is used).

    Pushes each expert towards ~1/K of the traffic ON AVERAGE; a single sample is
    still free to route sharply.
    """
    missing = ~AVAILABLE.to(modes.device)[modes]  # [B, 3]
    n_experts = probs.shape[-1]
    terms = []
    for j in range(3):
        sel = missing[:, j]
        if sel.any():
            mean_p = probs[sel, j].mean(dim=0)
            terms.append((mean_p * (mean_p * n_experts).clamp(min=EPS).log()).sum())
    return torch.stack(terms).mean() if terms else None
