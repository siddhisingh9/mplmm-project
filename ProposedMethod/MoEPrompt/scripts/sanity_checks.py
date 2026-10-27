"""Automated checks that the MoE model and losses are correct before training.

  1. parameters  : only moe* tensors are new; backbone frozen (1,061,280); counts
  2. equivalence : MoE off == baseline PromptModel, and MoE on with gate 0 ==
                   baseline, bit for bit, in all 7 modes; gate open differs
  3. router      : weights sum to 1 per sample and are exactly 1/K at init
  4. leakage     : outputs do not change when the content of the missing
                   modalities is replaced by noise (MoE on, gate open, trained-
                   looking random router) -> nothing reads a missing modality
  5. losses      : one (generated, real) pair per missing modality; real target
                   detached; teacher has no grad; all losses finite; with the
                   gate at 0 only the gate gets a gradient from the MoE branch,
                   once open the experts and router do; frozen weights get none
  6. real batch  : a training batch at drop rate 0.7 runs through every loss

    python scripts/sanity_checks.py --dataset mosi
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from moemm.bridge import PromptModel, transfer_model  # noqa: E402
from moemm.config import DATASETS, FULL_MODE, PRETRAINED_BACKBONE, audio_len, dataset_cfg  # noqa: E402
from moemm.data import load_split, make_loader, to_device  # noqa: E402
from moemm.experts import MISSING, MOD_INDEX  # noqa: E402
from moemm.losses import distillation_loss, load_balance_loss, reconstruction_loss  # noqa: E402
from moemm.model import MoEPromptModel, make_hyp_params  # noqa: E402

FAILURES = []


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def nonzero_like(model):
    """Give the zero-initialised prompts non-zero values, so that every path matters."""
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(0)  # MPLMM: CUDA
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("prompt") or name in ("generative_prompt", "missing_type_prompt",
                                                     "m_a", "m_v", "m_l"):
                p.copy_(0.1 * torch.randn(p.shape, generator=gen).to(p.device))


def check_parameters(hyp, device):
    base = transfer_model(PromptModel(hyp).to(device), str(PRETRAINED_BACKBONE))
    model = transfer_model(MoEPromptModel(hyp).to(device), str(PRETRAINED_BACKBONE))
    fresh = MoEPromptModel(hyp).to(device)
    missing, unexpected = fresh.load_state_dict(base.state_dict(), strict=False)
    report("only moe* tensors are new", bool(missing) and all(k.startswith("moe") for k in missing)
           and not unexpected, f"{len(missing)} new tensors")

    def count(m, trainable):
        return sum(p.numel() for p in m.parameters() if p.requires_grad == trainable)

    new = sum(p.numel() for p in model.moe_parameters())
    print(f"       baseline trainable {count(base, True):,} | MoE trainable {count(model, True):,} "
          f"(new moe* {new:,}) | frozen {count(model, False):,}")
    report("backbone stays frozen (1,061,280 parameters)", count(model, False) == 1_061_280,
           f"{count(model, False):,}")
    report("all moe* parameters are trainable", all(p.requires_grad for p in model.moe_parameters()))


def build_pair(hyp, device):
    """Baseline PromptModel and MoE model with identical shared weights, in eval mode."""
    torch.manual_seed(0)
    base = transfer_model(PromptModel(hyp).to(device), str(PRETRAINED_BACKBONE))
    nonzero_like(base)
    moe = MoEPromptModel(hyp).to(device)
    moe.load_state_dict(base.state_dict(), strict=False)
    off = MoEPromptModel(hyp, {"enabled": False}).to(device)
    off.load_state_dict(base.state_dict())
    return base.eval(), moe.eval(), off.eval()


def all_modes_batch(dataset, audio_len, device, n=35):
    data = load_split(dataset, "valid", full_data=True, audio_len=audio_len)
    (text, audio, vision), _, _ = to_device(next(iter(make_loader(data, batch_size=n))), device)
    modes = torch.arange(text.shape[0], device=device) % 7  # every missing mode, 5x each
    return text, audio, vision, modes


@torch.no_grad()
def check_equivalence_and_router(base, moe, off, batch):
    text, audio, vision, modes = batch
    out_base = base(text, audio, vision, modes)
    out_off = off(text, audio, vision, modes)
    out_closed, aux = moe(text, audio, vision, modes, return_aux=True)
    report("MoE off == baseline PromptModel (all 7 modes)", torch.equal(out_base, out_off),
           f"max diff {(out_base - out_off).abs().max().item():.2e}")
    report("MoE on, gate 0 == baseline PromptModel", torch.equal(out_base, out_closed),
           f"max diff {(out_base - out_closed).abs().max().item():.2e}")

    probs = aux["probs"]
    k = moe.moe_cfg["k"]
    report("router weights sum to 1", torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0])))
    report(f"router weights are exactly 1/{k} at init", torch.allclose(probs, torch.full_like(probs, 1 / k)))

    moe.moe_gate.fill_(1.0)
    out_open = moe(text, audio, vision, modes)
    changed = (out_open - out_base).abs().flatten(1).amax(1) > 0
    report("gate open: outputs change exactly for samples with a missing modality",
           torch.equal(changed, modes != FULL_MODE),
           f"{int(changed.sum())} changed, {int((modes != FULL_MODE).sum())} with missing modalities")


@torch.no_grad()
def check_leakage(moe, off, batch):
    """Replace the content of missing modalities by noise: outputs must not move."""
    text, audio, vision, modes = batch
    nn.init.normal_(moe.moe.router.mlp[-1].weight, std=1.0)  # make routing input-dependent
    moe.moe_gate.fill_(1.0)
    for name, model in (("MoE (gate open)", moe), ("baseline path", off)):
        worst = 0.0
        for mode in range(7):
            modes_m = torch.full_like(modes, mode)
            clean = model(text, audio, vision, modes_m)
            noisy = [text.clone(), audio.clone(), vision.clone()]
            for m in MISSING[mode]:
                j = MOD_INDEX[m]
                noisy[j] = 100 * torch.randn_like(noisy[j])
            out = model(*noisy, modes_m)
            worst = max(worst, (out - clean).abs().max().item())
        report(f"no leakage from missing modalities: {name}", worst == 0.0, f"max change {worst:.2e}")
    probs_a = moe(text, audio, vision, torch.full_like(modes, 0), return_aux=True)[1]["probs"]
    report("router is input dependent once trained-looking",
           probs_a.std(dim=0).max().item() > 1e-3)


def check_losses(moe, batch, train_batch, cfg):
    text, audio, vision, modes = batch
    moe.train()
    moe.moe_gate.data.zero_()
    nn.init.zeros_(moe.moe.router.mlp[-1].weight)

    preds, aux = moe(text, audio, vision, modes, return_aux=True)
    expected = sum(len(MISSING[int(m)]) for m in modes)
    report("one reconstruction pair per missing modality", len(aux["recon"]) == expected,
           f"{len(aux['recon'])} pairs, {expected} missing modalities")
    report("reconstruction targets are detached", all(not t.requires_grad for _, t in aux["recon"]))

    with torch.no_grad():
        moe.eval()
        _, teacher = moe(text, audio, vision, torch.full_like(modes, FULL_MODE), return_aux=True)
        moe.train()
    report("teacher pass: no gradient, no generated modalities",
           not teacher["features"].requires_grad and len(teacher["recon"]) == 0)

    def total_loss(preds, aux):
        task = (nn.CrossEntropyLoss()(preds.view(-1, cfg["output_dim"]), y_cls)
                if cfg["task"] == "classification" else preds.abs().mean())
        rec = reconstruction_loss(aux["recon"])
        kd = distillation_loss(aux["features"], teacher["features"], aux["modes"])
        bal = load_balance_loss(aux["probs"], aux["modes"])
        return task + rec + kd + 0.01 * bal, (rec, kd, bal)

    y_cls = torch.zeros(text.shape[0], dtype=torch.long, device=text.device)
    loss, (rec, kd, bal) = total_loss(preds, aux)
    report("losses finite", all(torch.isfinite(t).item() for t in (loss, rec, kd, bal)),
           f"rec {rec.item():.3f}  kd {kd.item():.3f}  bal {bal.item():.2e}")
    moe.zero_grad()
    loss.backward()
    # The router also sees the balance loss directly (not through the gate); at
    # uniform routing that gradient is 0 up to float error, so only experts are tested.
    gate_grad = moe.moe_gate.grad.abs().sum().item()
    expert_grad = sum(p.grad.abs().sum().item() for n, p in moe.named_parameters()
                      if n.startswith("moe.translators") and p.grad is not None)
    report("gate 0: gate gets gradient, experts get exactly none yet",
           gate_grad > 0 and expert_grad == 0, f"gate {gate_grad:.2e}, experts {expert_grad:.2e}")

    moe.moe_gate.data.fill_(0.5)
    moe.zero_grad()
    preds, aux = moe(text, audio, vision, modes, return_aux=True)
    total_loss(preds, aux)[0].backward()
    no_grad = [n for n, p in moe.named_parameters() if n.startswith("moe.translators") and
               (p.grad is None or p.grad.abs().sum() == 0)]
    report("gate open: every expert translator gets a gradient", not no_grad, f"{len(no_grad)} without")
    frozen_with_grad = [n for n, p in moe.named_parameters() if not p.requires_grad and p.grad is not None]
    report("frozen backbone gets no gradient", not frozen_with_grad, f"{len(frozen_with_grad)} with grad")

    # A real training batch, with MPLMM's own random missing modes.
    (t, a, v), y, m = train_batch
    preds, aux = moe(t, a, v, m, return_aux=True)
    rec = reconstruction_loss(aux["recon"])
    bal = load_balance_loss(aux["probs"], aux["modes"])
    report("real training batch runs through the losses",
           rec is not None and torch.isfinite(rec).item() and torch.isfinite(bal).item(),
           f"modes {torch.bincount(m, minlength=7).tolist()}, rec {rec.item():.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--audio_len", type=int, default=None,
                        help="IEMOCAP audio frames (default: dataset setting, 150)")
    args = parser.parse_args()
    args.audio_len = audio_len(args.dataset, args.audio_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dataset_cfg(args.dataset)
    train = load_split(args.dataset, "train", drop_rate=0.7, audio_len=args.audio_len)
    hyp = make_hyp_params(args.dataset, train)

    check_parameters(hyp, device)
    base, moe, off = build_pair(hyp, device)
    batch = all_modes_batch(args.dataset, args.audio_len, device)
    check_equivalence_and_router(base, moe, off, batch)
    check_leakage(moe, off, batch)
    train_batch = to_device(next(iter(make_loader(train, batch_size=32, shuffle=True))), device)
    check_losses(moe, batch, train_batch, cfg)

    print("\nALL CHECKS PASSED" if not FAILURES else f"\n{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
