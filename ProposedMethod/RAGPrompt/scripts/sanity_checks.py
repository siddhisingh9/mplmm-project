"""Automated checks that the RAG pipeline is correct before training on it.

  1. evaluation : ragmm.evaluate reproduces a baseline table row (optional)
  2. alignment  : bank entry i is training sample i (same label)
  3. leakage    : in training mode no sample retrieves itself or its own group
  4. equivalence: with the gate at 0 the RAG model gives EXACTLY the baseline
                  PromptModel output; with the gate open it differs
  5. parameters : trainable / frozen counts, and how many are new rag_* layers

    python scripts/sanity_checks.py --dataset sims \
        --legacy_ckpt ../MPLMM/results/sims_fixed/stage2_lr0.001_seed1111.pt --expect 74.58 70.48
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from ragmm.bank import MODE_SPEC, MemoryBank  # noqa: E402
from ragmm.bridge import PromptModel, transfer_model  # noqa: E402
from ragmm.config import DATASETS, PRETRAINED_BACKBONE, bank_path  # noqa: E402
from ragmm.data import load_split, make_loader, to_device  # noqa: E402
from ragmm.evaluate import evaluate_cases, format_results  # noqa: E402
from ragmm.model import RAGPromptModel, make_hyp_params  # noqa: E402

FAILURES = []


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def check_evaluation(args, device):
    model = torch.load(args.legacy_ckpt, map_location=device, weights_only=False).to(device)
    results = evaluate_cases(model, args.dataset, "test", device, audio_len=args.audio_len)
    print(format_results(results))
    acc, f1 = results["Avg."]["ACC"], results["Avg."]["F1"]
    ok = abs(acc - args.expect[0]) < 0.01 and abs(f1 - args.expect[1]) < 0.01
    report("evaluation reproduces the baseline table", ok,
           f"got {acc:.2f}/{f1:.2f}, expected {args.expect[0]:.2f}/{args.expect[1]:.2f}")


def check_alignment_and_leakage(args, bank, device):
    train = load_split(args.dataset, "train", drop_rate=0.0, full_data=True, audio_len=args.audio_len)
    report("bank size equals training split", len(bank) == len(train), f"{len(bank)} vs {len(train)}")

    loader = make_loader(train, batch_size=64, shuffle=True, seed=0, with_index=True)
    aligned, leaks, self_hits, self_hits_idx = True, 0, 0, 0
    for step, batch in enumerate(loader):
        (text, audio, vision), y, _, idx = to_device(batch, device)
        target = y.reshape(-1).to(bank.labels.dtype)
        aligned &= torch.equal(bank.labels[idx], target)
        queries = bank.query_keys(text, audio, vision)
        own_groups = bank.groups[idx]
        for available, _ in MODE_SPEC.values():
            got, _ = bank.retrieve(queries, available, k=10, tau=0.1, exclude_groups=own_groups)
            leaks += (bank.groups[got] == own_groups.unsqueeze(1)).sum().item()
            self_hits += (got == idx.unsqueeze(1)).sum().item()
            got, _ = bank.retrieve(queries, available, k=10, tau=0.1, exclude_indices=idx)
            self_hits_idx += (got == idx.unsqueeze(1)).sum().item()
        if step == 4:
            break
    report("bank entry i is training sample i (labels match)", aligned)
    report("no retrieval from the sample's own video/dialogue", leaks == 0, f"{leaks} leaks")
    report("no sample retrieves itself", self_hits == 0, f"{self_hits} self-hits")
    report("--exclude self: no sample retrieves itself", self_hits_idx == 0,
           f"{self_hits_idx} self-hits")


def check_equivalence(args, bank, device):
    train = load_split(args.dataset, "train", audio_len=args.audio_len)
    hyp = make_hyp_params(args.dataset, train)

    torch.manual_seed(0)
    baseline = transfer_model(PromptModel(hyp).to(device), str(PRETRAINED_BACKBONE)).eval()
    rag = RAGPromptModel(hyp, {"enabled": True}).to(device)
    missing, unexpected = rag.load_state_dict(baseline.state_dict(), strict=False)
    report("only rag_* parameters are new", all(k.startswith("rag_") for k in missing) and not unexpected,
           f"{len(missing)} new tensors")
    rag.attach_bank(bank).eval()
    static = RAGPromptModel(hyp, {"enabled": False}).to(device)
    static.load_state_dict(baseline.state_dict(), strict=False)
    static.eval()

    # Non-zero prompts, so that an open gate visibly changes the output.
    with torch.no_grad():
        for m in (baseline, rag, static):
            m.generative_prompt.copy_(torch.linspace(-1, 1, baseline.generative_prompt.numel())
                                      .view_as(baseline.generative_prompt))

    valid = load_split(args.dataset, "valid", full_data=True, audio_len=args.audio_len)
    (text, audio, vision), _, _ = to_device(next(iter(make_loader(valid, batch_size=32))), device)
    modes = torch.arange(text.shape[0], device=device) % 7  # every missing mode
    with torch.no_grad():
        out_base = baseline(text, audio, vision, modes)
        out_static = static(text, audio, vision, modes)
        out_rag_closed = rag(text, audio, vision, modes)
        rag.rag_gate.fill_(1.0)
        out_rag_open = rag(text, audio, vision, modes)

    report("static (RAG off) == baseline PromptModel", torch.equal(out_base, out_static),
           f"max diff {(out_base - out_static).abs().max().item():.2e}")
    report("RAG with gate 0 == baseline PromptModel", torch.equal(out_base, out_rag_closed),
           f"max diff {(out_base - out_rag_closed).abs().max().item():.2e}")
    report("RAG with gate open != baseline (retrieval path is live)",
           not torch.allclose(out_base, out_rag_open),
           f"max diff {(out_base - out_rag_open).abs().max().item():.2e}")
    return hyp


def check_parameters(args, hyp, device):
    model = RAGPromptModel(hyp, {"enabled": True}).to(device)
    model = transfer_model(model, str(PRETRAINED_BACKBONE))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    new = sum(p.numel() for p in model.rag_parameters())
    print(f"       trainable {trainable:,}  (of which new rag_* {new:,})  frozen {frozen:,}")
    report("all rag_* parameters are trainable", all(p.requires_grad for p in model.rag_parameters()))
    report("backbone stays frozen (1,061,280 parameters)", frozen == 1_061_280, f"{frozen:,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--audio_len", type=int, default=350)
    parser.add_argument("--legacy_ckpt", type=Path, help="baseline checkpoint to re-evaluate")
    parser.add_argument("--expect", type=float, nargs=2, metavar=("ACC", "F1"),
                        help="expected Avg. ACC and F1 for --legacy_ckpt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = MemoryBank.load(bank_path(args.dataset, args.audio_len), device)

    if args.legacy_ckpt:
        check_evaluation(args, device)
    check_alignment_and_leakage(args, bank, device)
    hyp = check_equivalence(args, bank, device)
    check_parameters(args, hyp, device)

    print("\nALL CHECKS PASSED" if not FAILURES else f"\n{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
