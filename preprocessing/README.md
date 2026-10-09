# Pre-processing

Feature extraction and exploratory data analysis (EDA) for **CH-SIMS**, **CMU-MOSI** and **CMU-MOSEI**.
(IEMOCAP is covered in [../notebooks/](../notebooks/).)

---

## Notebooks

Each dataset has two notebooks: **RAW** builds features from the original videos, and **PROCESSED** explores the standard released features.

| # | Notebook | Dataset | What it does |
|---|---|---|---|
| 01 | [01_CHSIMS_RAW_FINAL.ipynb](01_CHSIMS_RAW_FINAL.ipynb) | CH-SIMS | Raw videos → EDA → T/A/V features → final pickle |
| 02 | [02_CHSIMS_PROCESSED_FINAL.ipynb](02_CHSIMS_PROCESSED_FINAL.ipynb) | CH-SIMS | EDA on processed unaligned features + Ridge T/A/V baseline |
| 03 | [03_CMUMOSI_RAW_FINAL.ipynb](03_CMUMOSI_RAW_FINAL.ipynb) | CMU-MOSI | Raw videos → EDA → T/A/V features → final pickle |
| 04 | [04_CMUMOSI_PROCESSED_FINAL.ipynb](04_CMUMOSI_PROCESSED_FINAL.ipynb) | CMU-MOSI | EDA on processed features, aligned vs unaligned |
| 05 | [05_CMUMOSEI_RAW_FINAL.ipynb](05_CMUMOSEI_RAW_FINAL.ipynb) | CMU-MOSEI | Raw videos → EDA → T/A/V features → final pickle |
| 06 | [06_CMUMOSEI_PROCESSED_FINAL.ipynb](06_CMUMOSEI_PROCESSED_FINAL.ipynb) | CMU-MOSEI | EDA on processed features, aligned vs unaligned |

---

## RAW Pipeline (01, 03, 05)

```text
Raw videos + labels → match sample IDs → EDA → extract T / A / V → final pickle → sanity check
```

| Modality | Extractor | Output |
|---|---|---|
| **Text** | BERT `[CLS]` embedding (`bert-base-uncased`; `bert-base-chinese` for CH-SIMS) | 768-d |
| **Audio** | ffmpeg → openSMILE `eGeMAPSv02` functionals | 88-d |
| **Visual** | ResNet-18 (ImageNet) avg-pool, mean over 5 uniformly sampled frames | 512-d |

Features are cached per modality, so extraction can resume after interruption.

---

## PROCESSED EDA (02, 04, 06)

- Structure: samples, keys, shapes, dtypes per split
- Labels: sentiment distribution, polarity balance, split comparison
- Feature quality: NaN / Inf / zero-vector checks, norms, IQR outliers
- Structure: PCA of text, audio and visual features
- Quick baseline: Ridge regression on unimodal and fused T/A/V features
- Aligned vs unaligned comparison (MOSI, MOSEI)

---

## Outputs

```text
outputs/<DATASET>/
├── artifacts/        # Extracted feature pickles and per-modality caches
├── figures/          # EDA plots
├── tables/           # EDA statistics (CSV)
├── model/            # Quick-baseline model and metrics
└── run_summary.json  # Split sizes and source paths
```

---

## Running

The notebooks run on **Google Colab** with data on Google Drive. Update the paths in the configuration cell to your own layout before running.
