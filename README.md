# Spatially-Enriched, Retrieval-Augmented and Expert-Specialized Missing-Modality Generation for Multimodal Sentiment Analysis

> Course project — Machine Learning, Semester 5, **Indian Institute of Technology Bhilai**

We extend **MPLMM** (*Multimodal Prompt Learning with Missing Modalities*, Guo et al., ACL 2024) so that sentiment analysis stays robust when text, audio or video is missing.

---

## Datasets

| Dataset | Language | Task |
|---|---|---|
| **CMU-MOSI** | English | Sentiment analysis |
| **CMU-MOSEI** | English | Sentiment analysis |
| **CH-SIMS** | Chinese | Sentiment analysis |
| **IEMOCAP** | English | Emotion recognition |

---

## Where We Are

**All four datasets are pre-processed and explored, and the MPLMM baseline is reproduced. Next, we add the three proposed techniques.**

| Step | Task | Status |
|---|---|---|
| 1 | Pre-processing and EDA: CMU-MOSI, CMU-MOSEI, CH-SIMS, IEMOCAP | ✅ Done |
| 2 | MPLMM baseline reproduction (pre-training and fine-tuning) | ✅ Done |
| 3 | Technique A: 2D spatial feature convolution | 🔲 Next |
| 4 | Technique B: Retrieval-augmented reconstruction (RAG, with Residual RAG as an extension) | 🔲 Next |
| 5 | Technique C: Mixture-of-Experts missing-modality generator (MoE-MMGM) | 🔲 Next |

---

## Repository Structure

```text
.
├── preprocessing/   # Feature extraction + EDA for CH-SIMS, CMU-MOSI, CMU-MOSEI
├── notebooks/       # IEMOCAP EDA + per-dataset quick classifiers
├── Baseline/        # MPLMM baseline reproduction (code, notebook, results)
├── configs/         # Experiment configuration
└── proposal.pdf     # Full research plan
```

- **Pre-processing:** see [preprocessing/README.md](preprocessing/README.md).
- **Notebooks:** see [notebooks/README.md](notebooks/README.md).
- **Baseline:** see [Baseline/README.md](Baseline/README.md) for setup, pre-training and fine-tuning commands. Results are in [Baseline/MPLMM/results/](Baseline/MPLMM/results/).

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
