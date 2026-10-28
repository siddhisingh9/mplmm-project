# Spatially-Enriched, Retrieval-Augmented and Expert-Specialized Missing-Modality Generation for Multimodal Sentiment Analysis

> Course project — Machine Learning, Semester 5, **Indian Institute of Technology Bhilai**

We extend **MPLMM** (*Multimodal Prompt Learning with Missing Modalities*, Guo et al., ACL 2024) so that sentiment analysis and emotion recognition stay robust when text, audio or video is missing.

**Quick navigation:** [Problem](#problem-statement) · [Datasets](#datasets) · [Progress](#progress) · [Repository](#repository-structure) · [Where to find what](#where-to-find-what) · [Team](#team) · [References](#references)

---

## Problem Statement

**Why it matters.** People express sentiment and emotion through *what* they say (text), *how* they say it (audio) and *how they look* (video). Multimodal models combine all three, but real inputs are often incomplete: the microphone fails, the face is hidden, or there is no transcript. Most models are trained on complete data, so their accuracy drops sharply when a modality is missing at test time.

**The problem.** An utterance has three modalities, $x = \{x_T, x_A, x_V\}$, and any one or two of them may be missing. Given only the available modalities $x_{\text{avail}}$, we want to:

1. **Reconstruct** each missing modality, $\hat{x}_m = G(x_{\text{avail}})$.
2. **Predict** sentiment or emotion from the available and reconstructed modalities.

The model must stay accurate across all missing-modality patterns and missing rates.

**Where MPLMM falls short.** MPLMM ([Guo et al., 2024](#references)) reconstructs missing modalities with learnable prompts, but:

- Its visual features are flat vectors, so the spatial structure of the face is lost.
- Each sample is reconstructed alone, without using similar complete examples from the training data.
- A single generator must handle every missing-modality pattern.

**Our vision.** We build a missing-modality generator that keeps facial detail (**2D spatial features**), learns from similar complete training examples (**retrieval augmentation**) and uses specialised experts for different missing patterns (**MoE-MMGM**). The goal is robust sentiment analysis and emotion recognition that degrades gracefully as modalities go missing, tested on four benchmarks in English and Chinese.

---

## Datasets

| Dataset | Language | Task |
|---|---|---|
| **CMU-MOSI** | English | Sentiment analysis |
| **CMU-MOSEI** | English | Sentiment analysis |
| **CH-SIMS** | Chinese | Sentiment analysis |
| **IEMOCAP** | English | Emotion recognition |

---

## Progress

| | Stage | Status |
|:-:|---|---|
| ✅ | Pre-processing and EDA on all 4 datasets | Done |
| ✅ | MPLMM baseline reproduction (pre-training and fine-tuning) | Done |
| ✅ | **Technique A:** 2D spatial feature convolution | Partial: 2D-CNN (ResNet-18) features extracted from raw frames; tested on sentiment analysis with all modalities present; not used in the missing-modality pipeline |
| ✅ | **Technique B:** Retrieval-augmented generation (RAGPrompt) | Implemented and evaluated on all 4 datasets |
| ✅ | **Technique C:** Mixture-of-Experts generator (MoEPrompt) | Implemented and evaluated on all 4 datasets |

| ⬜ | **Residual RAG** (exploratory extension of B) | Not started |

---

## Repository Structure

```text
.
├── preprocessing/       # Feature extraction + EDA for CH-SIMS, CMU-MOSI, CMU-MOSEI
├── notebooks/           # IEMOCAP EDA + per-dataset quick classifiers
├── Baseline/            # MPLMM baseline reproduction (code, notebook, results)
├── ProposedMethod/
│   ├── MPLMM/           # Read-only copy of the baseline code, imported by both extensions
│   ├── RAGPrompt/       # Technique B: retrieval-augmented generative prompts
│   ├── MoEPrompt/       # Technique C: mixture-of-experts missing-modality generator
│   └── finalResults.ipynb  # Final comparison tables (Paper vs Baseline vs RAG vs MoE)
├── LICENSE
└── proposal.pdf         # Full research plan
```

---

## Where to Find What

| I want to… | Go to |
|---|---|
| See the final comparison tables | [ProposedMethod/finalResults.ipynb](ProposedMethod/finalResults.ipynb) |
| Understand the baseline and reproduce it | [Baseline/README.md](Baseline/README.md) |
| Understand RAG and run it | [ProposedMethod/RAGPrompt/README.md](ProposedMethod/RAGPrompt/README.md) |
| Understand MoE and run it | [ProposedMethod/MoEPrompt/README.md](ProposedMethod/MoEPrompt/README.md) |
| See the baseline result table | [Baseline/MPLMM/results/mplmm_baseline_results.csv](Baseline/MPLMM/results/mplmm_baseline_results.csv) |
| See the dataset pre-processing | [preprocessing/README.md](preprocessing/README.md) |
| See the IEMOCAP and classifier notebooks | [notebooks/README.md](notebooks/README.md) |

---

## Team

| Member | Roll No. | Responsibilities |
|---|---|---|
| Lanka Devi Satwika | B24DS013 | MPLMM reproduction, dataset preparation, baseline evaluation |
| Siddhi Singh | B24DS508 | 2D spatial feature convolution, visual pre-processing, spatial-feature evaluation |
| Vidit Shrimali | B24DS038 | Retrieval system, Residual RAG, MoE-MMGM, ablation studies, overall evaluation |

---

## References

1. Z. Guo, T. Jin, and Z. Zhao, "Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition," *Proc. ACL*, 2024.
2. Y.-H. H. Tsai et al., "Multimodal Transformer for Unaligned Multimodal Language Sequences," *Proc. ACL*, 2019.
3. A. Zadeh et al., "MOSI: Multimodal Corpus of Sentiment Intensity and Subjectivity Analysis in Online Opinion Videos," *arXiv:1606.06259*, 2016.
4. A. Zadeh et al., "Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph," *Proc. ACL*, 2018.
5. C. Busso et al., "IEMOCAP: Interactive Emotional Dyadic Motion Capture Database," *Language Resources and Evaluation*, 2008.
6. W. Yu et al., "CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality," *Proc. ACL*, 2020.

---

## License

Released under the [MIT License](LICENSE). © 2026 Siddhi Singh, Vidit Shrimali, Lanka Devi Satwika.

The datasets (CMU-MOSI, CMU-MOSEI, IEMOCAP, CH-SIMS) and pretrained models used here are subject to their own licenses.
