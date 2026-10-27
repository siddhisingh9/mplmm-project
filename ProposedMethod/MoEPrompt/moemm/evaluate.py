"""Six-case evaluation with the same conventions as the final baseline table.

  regression (MOSI, MOSEI, CH-SIMS): sign of the score -> binary label,
      positive iff >= 0 (MOSI, MOSEI) or > 0 (CH-SIMS)
  classification (IEMOCAP): argmax over the 4 classes
  ACC = accuracy, F1 = weighted F1, both in %. Avg. = mean of the 6 cases.

routing_table() reports how the router splits each missing modality over the
experts, per case.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from moemm.config import CASES, dataset_cfg
from moemm.data import load_split, make_loader, to_device
from moemm.experts import MISSING, MOD_INDEX

MOD_NAME = {"l": "t", "a": "a", "v": "v"}  # model letter -> table letter


def to_labels(dataset, outputs):
    """Model outputs / targets (numpy) -> discrete labels."""
    cfg = dataset_cfg(dataset)
    if cfg["task"] == "classification":
        return outputs.argmax(axis=1) if outputs.ndim == 2 else outputs.astype(int)
    return outputs >= 0 if cfg["positive"] == ">=0" else outputs > 0


@torch.no_grad()
def predict(model, loader, mode, device, with_probs=False):
    """Run every sample with the same forced missing mode."""
    model.eval()
    outputs, targets, probs = [], [], []
    for batch in loader:
        (text, audio, vision), y, _ = to_device(batch, device)
        modes = torch.full((text.shape[0],), mode, dtype=torch.long, device=device)
        if with_probs:
            out, aux = model(text, audio, vision, modes, return_aux=True)
            probs.append(aux["probs"].cpu())
        else:
            out = model(text, audio, vision, modes)
        outputs.append(out.float().cpu())
        targets.append(y.reshape(-1).cpu())
    result = torch.cat(outputs).numpy(), torch.cat(targets).numpy()
    return result + (torch.cat(probs),) if with_probs else result


def evaluate_cases(model, dataset, split, device, audio_len=350, batch_size=64):
    """{case: {"ACC", "F1"}} for the 6 missing-modality cases plus "Avg."."""
    cfg = dataset_cfg(dataset)
    data = load_split(dataset, split, full_data=True, audio_len=audio_len)
    loader = make_loader(data, batch_size=batch_size)  # not shuffled: dataset order

    results = {}
    for case, mode in CASES.items():
        outputs, targets = predict(model, loader, mode, device)
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


def routing_table(model, dataset, split, device, audio_len=350, batch_size=64):
    """Rows (case, generated modality): mean router weight per expert, mean
    per-sample entropy (0 = always one expert, log K = uniform) and the share of
    samples whose largest weight is each expert."""
    data = load_split(dataset, split, full_data=True, audio_len=audio_len)
    loader = make_loader(data, batch_size=batch_size)
    rows = []
    for case, mode in CASES.items():
        _, _, probs = predict(model, loader, mode, device, with_probs=True)
        for m in MISSING[mode]:
            p = probs[:, MOD_INDEX[m]]  # [N, K]
            row = {"case": case, "generated": MOD_NAME[m]}
            row.update({f"mean p{k + 1}": round(v, 3) for k, v in enumerate(p.mean(0).tolist())})
            row["entropy"] = round(float(-(p * p.clamp(min=1e-12).log()).sum(1).mean()), 3)
            top = torch.bincount(p.argmax(1), minlength=p.shape[1]).float() / len(p)
            row.update({f"top{k + 1} share": round(v, 3) for k, v in enumerate(top.tolist())})
            rows.append(row)
    return rows


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
