# Notebooks

IEMOCAP pre-processing and EDA, plus a quick end-to-end classifier for each dataset.

---

## Notebooks

| Notebook | Dataset | What it does |
|---|---|---|
| [IEMOCAP_EDA_Preprocessing_(ALIGN).ipynb](IEMOCAP_EDA_Preprocessing_(ALIGN).ipynb) | IEMOCAP | EDA, modality probes and cleaning on **aligned** features |
| [IEMOCAP_EDA_Preprocessing(NOALIGN) (1).ipynb](<IEMOCAP_EDA_Preprocessing(NOALIGN) (1).ipynb>) | IEMOCAP | Same analysis on **unaligned** features |
| [IEMOCAP_Classifier.ipynb](IEMOCAP_Classifier.ipynb) | IEMOCAP | EDA + logistic-regression emotion classifier + inference |
| [CMU_MOSI_Classifier.ipynb](CMU_MOSI_Classifier.ipynb) | CMU-MOSI | Raw feature extraction + MulT classifier + raw-video inference |
| [CH_SIMS_Classifier.ipynb](CH_SIMS_Classifier.ipynb) | CH-SIMS | Raw-video EDA + MulT sentiment regressor |

---

## Techniques

### 1. Feature Extraction (CMU-MOSI)

Features come from the **raw** CMU-MOSI release (93 videos, 2,199 utterances), not the standard pre-computed SDK features.

| Modality | Extractor | Details | Dim |
|---|---|---|---|
| **Text** | `bert-base-uncased` | `[CLS]` token of the last hidden state; max 64 tokens | 768 |
| **Acoustic** | openSMILE `eGeMAPSv02` functionals | 88 functionals truncated to 74, matching MPLMM's COVAREP dimension | 74 |
| **Visual** | ResNet-18 (ImageNet) | 5 uniform frames → 224×224 → 512-d → mean → linear 512 → 35, matching FACET | 35 |

CH-SIMS uses `bert-base-chinese` (text) and `chinese-wav2vec2-base` (audio).

### 2. Classifier

- **MulT-style crossmodal transformer:** six crossmodal blocks (T↔A, T↔V, A↔V) followed by self-attention per modality.
- **CMU-MOSI:** per-utterance binary classification (up to 63 utterances per video), AdamW + cosine LR, early stopping.
- **CH-SIMS:** continuous sentiment regression (MSE) on processed unaligned features.
- **IEMOCAP:** logistic regression on pooled features as a quick emotion baseline.

### 3. Missing-Modality Analysis (IEMOCAP)

- **Modality probes:** unimodal, bimodal and trimodal linear probes show which modality matters most.
- **Reconstructability:** Ridge regression predicts a missing modality from the available ones (R²).
- **RAG feasibility:** kNN retrieval and residual correction on pooled features, an early check for Techniques B and Residual RAG.
- **Simulator:** a PyTorch `Dataset` that drops modalities to simulate missing inputs.

### 4. Inference on Raw Video (CMU-MOSI)

Whisper (`small`) transcribes the speech, the same extractors produce T/A/V features, and the saved MulT model ([../configs/config.pkl](../configs/config.pkl)) predicts sentiment.

---

## Running

Notebooks run on **Google Colab** or **Kaggle**. Update the dataset paths in the setup cells before running.
