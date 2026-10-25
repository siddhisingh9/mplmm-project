# RAGPrompt: retrieval-augmented generative prompts for MPLMM

Extension of the MPLMM baseline (`../MPLMM`, used read-only). In MPLMM the
**generative prompt** that helps reconstruct a missing modality is one learned
tensor shared by every sample. Here it is built **per sample** from the most
similar training samples.

## Idea in one picture

```
sample x, modality m missing
  │
  ├─ available modalities ──► query keys ──► cosine top-K over the TRAINING bank
  │                                            (own video excluded while training)
  │                                                     │
  │                         retrieved: modality m of those K samples (16 segments each)
  │                                                     │
  └─ static prompt P_G[m] ──► cross-attention (queries = P_G[m], keys/values = retrieved,
                              biased by retrieval weights)
                                                        │
                    P_G(x)[m] = P_G[m] + tanh(gate_m) · attention output    (gate_m starts at 0)
                                                        │
                     rest of MPLMM unchanged (frozen MOSEI backbone, same prompts and heads)
```

At initialisation the model is exactly the baseline (gate = 0); training
decides how much of the retrieved information to use.

## Layout

| Path | Purpose |
|---|---|
| `ragmm/config.py` | paths, dataset settings, missing-modality cases |
| `ragmm/bridge.py` | the only file that imports MPLMM code |
| `ragmm/data.py` | MPLMM loaders + group ids (video/dialogue) + sample indices |
| `ragmm/bank.py` | memory bank: build, save/load, top-K retrieval |
| `ragmm/model.py` | `RAGPromptModel(PromptModel)`, checkpoint save/load |
| `ragmm/evaluate.py` | 6-case ACC / weighted F1, same conventions as the baseline table |
| `scripts/build_bank.py` | build the bank for one dataset |
| `scripts/sanity_checks.py` | correctness checks (must print ALL CHECKS PASSED) |
| `scripts/check_retrieval.py` | does retrieval carry signal? real vs random, no training |
| `scripts/train_rag.py` | train `static` / `rag` / `random`, append a row to `results/runs.csv` |
| `RAGExperiments.ipynb` | runs the scripts and builds the comparison table |

Generated (gitignored): `banks/`, `checkpoints/`.

## Run

```bash
cd RAGPrompt
python scripts/build_bank.py     --dataset sims
python scripts/sanity_checks.py  --dataset sims
python scripts/check_retrieval.py --dataset sims
python scripts/train_rag.py      --dataset sims --mode static
python scripts/train_rag.py      --dataset sims --mode rag --k 5
python scripts/train_rag.py      --dataset sims --mode random --k 5
```

No dependencies beyond the baseline's (torch, numpy, scikit-learn, pandas, h5py).

## Design decisions

* **Bank = training split only**, stored as raw features (keys: time-mean,
  standardised, L2-normalised; values: 16 time segments). It never changes
  during training, so retrieval is deterministic and needs no refresh.
* **Retrieval compares only the available modalities** with the same
  modalities in the bank, so no cross-modal encoder is needed.
* **Leakage**: while training, a sample never retrieves itself or a clip of
  its own video/dialogue (`--exclude group`, default). Validation/test samples
  are never in the bank.
* **CH-SIMS caveat**: its official split puts clips of the same video in
  train/valid/test, so test-time retrieval may return same-video clips. Every
  RAG run on CH-SIMS therefore also reports a **strict** test score that
  forbids this (`strict test ...` columns in `runs.csv`).
* **Fair comparison**: `static` is retrained through the same script (same
  shuffling, seed, schedule); `random` has the same extra parameters as `rag`
  but random neighbours, so a gain of `rag` over `random` is due to retrieval.
* **Selection**: best checkpoint by validation loss; K chosen on validation
  Avg. ACC; the test split is only evaluated after training.

## Verified so far

* evaluation reproduces the baseline table (CH-SIMS 74.58/70.48, MOSI 62.22/58.23);
* 0 leaks / 0 self-retrievals; bank entry i is training sample i;
* RAG with gate 0 == baseline `PromptModel`, bit for bit;
* retrieval check (validation, K=5): the retrieved missing modality matches
  the true one far better than random (cosine 0.14–0.60 vs ≈0), also under the
  strict CH-SIMS protocol.
