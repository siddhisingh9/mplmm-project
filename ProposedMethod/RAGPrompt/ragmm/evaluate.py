"""Six-case evaluation with the same conventions as the final baseline table.

  regression (MOSI, MOSEI, CH-SIMS): sign of the score -> binary label,
      positive iff >= 0 (MOSI, MOSEI) or > 0 (CH-SIMS)
  classification (IEMOCAP): argmax over the 4 classes
  ACC = accuracy, F1 = weighted F1, both in %. Avg. = mean of the 6 cases.

strict=True (RAG models only) additionally forbids retrieving training clips
from the evaluated clip's own video/dialogue. It only matters for CH-SIMS,
whose official split puts clips of the same video in train, valid and test.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from ragmm.config import CASES, dataset_cfg
from ragmm.data import group_names, load_split, make_loader, to_device


def to_labels(dataset, outputs):
    """Model outputs / targets (numpy) -> discrete labels."""
    cfg = dataset_cfg(dataset)
    if cfg["task"] == "classification":
        return outputs.argmax(axis=1) if outputs.ndim == 2 else outputs.astype(int)
    return outputs >= 0 if cfg["positive"] == ">=0" else outputs > 0


@torch.no_grad()
def predict(model, loader, mode, device, sample_groups=None):
    """Run every sample with the same forced missing mode.

    sample_groups: [N] bank group id per sample (strict mode), in loader order.
    """
    model.eval()
    outputs, targets, offset = [], [], 0
    for batch in loader:
        (text, audio, vision), y, _ = to_device(batch, device)
        b = text.shape[0]
        modes = torch.full((b,), mode, dtype=torch.long, device=device)
        if sample_groups is None:
            out = model(text, audio, vision, modes)
        else:
            out = model(text, audio, vision, modes, exclude_groups=sample_groups[offset:offset + b])
        outputs.append(out.float().cpu())
        targets.append(y.reshape(-1).cpu())
        offset += b
    return torch.cat(outputs).numpy(), torch.cat(targets).numpy()


def shares_groups_with_bank(bank, dataset, split):
    return bool((bank.group_ids(group_names(dataset, split)) >= 0).any())


def evaluate_cases(model, dataset, split, device, audio_len=350, batch_size=64,
                   strict=False, bank=None):
    """{case: {"ACC", "F1"}} for the 6 missing-modality cases plus "Avg."."""
    cfg = dataset_cfg(dataset)
    data = load_split(dataset, split, full_data=True, audio_len=audio_len)
    loader = make_loader(data, batch_size=batch_size)  # not shuffled: dataset order
    sample_groups = bank.group_ids(group_names(dataset, split)) if strict else None

    results = {}
    for case, mode in CASES.items():
        outputs, targets = predict(model, loader, mode, device, sample_groups)
        if cfg["task"] == "classification":
            outputs = outputs.reshape(-1, cfg["output_dim"])
        else:
            outputs = outputs.reshape(-1)
        y_true, y_pred = to_labels(dataset, targets), to_labels(dataset, outputs)
        results[case] = {
            "ACC": accuracy_score(y_true, y_pred) * 100,
            "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        }
    results["Avg."] = {
        metric: float(np.mean([results[c][metric] for c in CASES])) for metric in ("ACC", "F1")
    }
    return results


def flatten(results, prefix=""):
    """{case: {ACC, F1}} -> {"<prefix>{a} ACC": ..., ...} for CSV rows."""
    return {
        f"{prefix}{case} {metric}": round(values[metric], 2)
        for case, values in results.items()
        for metric in ("ACC", "F1")
    }


def format_results(results):
    cases = list(CASES) + ["Avg."]
    header = "".join(f"{c:>16s}" for c in cases)
    row = "".join(f"{results[c]['ACC']:8.2f}/{results[c]['F1']:6.2f} " for c in cases)
    return f"{'':6s}{header}\n{'ACC/F1':6s}{row}"
