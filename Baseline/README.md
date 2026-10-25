# Baseline

Reproduction workspace for the MPLMM multimodal learning experiments, including
training code, diagnostics, and notebook-based Figure 4 and Figure 6 analysis.

## Repository contents

- [`MPLMM/`](MPLMM/) - model, dataset-loader, training, and evaluation code
- [`BaselineReproduction.ipynb`](BaselineReproduction.ipynb) - experiment and
  plotting notebook
- [`MPLMM/results/`](MPLMM/results/) - generated figures when produced locally

Datasets and trained checkpoints are intentionally excluded from Git because
they are large and may have separate distribution terms. Place local datasets
under `MPLMM/dataset/` and local checkpoints under `MPLMM/pretrained/` or
`MPLMM/results/`.

## Environment

Use Python 3.8 or newer with PyTorch. Install the dependencies required by the
source code in the same environment used to run the notebook or commands. For
example:

```bash
cd MPLMM
python -m pip install torch numpy scipy scikit-learn h5py
```

When running from a notebook, use the notebook interpreter:

```python
import sys
!{sys.executable} -m pip install torch numpy scipy scikit-learn h5py
```

## Dataset paths

The notebook and commands expect these paths relative to `MPLMM/`:

```text
MPLMM/dataset/mosi_data.pkl
MPLMM/dataset/mosei_senti_data.pkl
MPLMM/dataset/sims_data.pkl
MPLMM/dataset/iemocap/IEMOCAP_features_2021/
```

## Pre Training

From the `MPLMM` directory, use the same Python interpreter as the notebook:

```bash
python main.py \
  --dataset mosei \
  --data_path dataset/mosei_senti_data.pkl \
  --drop_rate 0.0 \
  --num_epochs 30 \
  --name pretrained/mosei.pt
```

For downstream transfer, provide the pretrained MOSEI checkpoint with
`--pretrained_model pretrained/mosei.pt`.


## Fine Tuning

Fine-tune each dataset from the same `MPLMM/` directory using a uniform command
layout:

### CMU-MOSEI

```bash
python main.py \
  --pretrained_model pretrained/mosei.pt \
  --dataset mosei \
  --data_path dataset/mosei_senti_data.pkl \
  --drop_rate 0.7 \
  --num_epochs 30 \
  --name results/mosei_stage2.pt
```

### CMU-MOSI

```bash
python main.py \
  --pretrained_model pretrained/mosei.pt \
  --dataset mosi \
  --data_path dataset/mosi_data.pkl \
  --drop_rate 0.7 \
  --num_epochs 30 \
  --name results/mosi_stage2.pt
```

### IEMOCAP

```bash
python main.py \
  --pretrained_model pretrained/mosei.pt \
  --dataset iemocap \
  --data_path dataset/iemocap/IEMOCAP_features_2021 \
  --drop_rate 0.7 \
  --num_epochs 30 \
  --name results/iemocap_stage2.pt
```

### CH-SIMS

```bash
python main.py \
  --pretrained_model pretrained/mosei.pt \
  --dataset sims \
  --data_path dataset/sims_data.pkl \
  --drop_rate 0.7 \
  --num_epochs 30 \
  --name results/sims_stage2.pt
```

### Fine-tuning flags 

For the improved downstream runs we report for CMU-MOSI and IEMOCAP, we kept
the same model and transfer setup but additionally used:

```bash
--shuffle_train --selection_metric avg_f1
```

- `--shuffle_train` shuffles training batches each epoch.
- `--selection_metric avg_f1` saves the checkpoint with the best average
  validation F1 across the six fixed missing-modality cases instead of using
  validation loss alone.

## Figures

The notebook contains the deterministic evaluation and plotting cells for the
missing-modality experiments. Generated Figure 4 files are saved as:

```text
MPLMM/results/CMU MOSI Performance During Training.png
MPLMM/results/MPLMM Performance Under Different Test-Time Missing Rates.png
MPLMM/results/table.png
```

## Technique A: 2D spatial feature convolution (CMU-MOSI)

The MOSI vision features (FACET, 20-d per word) are replaced by face crops from
the raw videos. A frozen face-expression CNN turns each of 16 uniformly sampled
frames into a 7×7 feature map, keeping the spatial layout. A learnable 2D
convolution then projects each map to the visual sequence MPLMM uses
(proposal Eqs. 3–4, [`MPLMM/src/spatial.py`](MPLMM/src/spatial.py)):
BatchNorm → Conv2d 3×3 (32 ch) → ReLU → flatten → Linear (64) → LayerNorm.
All flags are off by default, so baseline runs are unchanged.

**Backbone.** The final model uses the HSEmotion EfficientNet-B0
(`enet_b0_8_va_mtl`, trained on AffectNet for facial expressions plus
valence/arousal), taking its 1280×7×7 head map, the map its expression and
valence classifier pools. Without any training, its predicted valence on MOSI
faces already correlates with sentiment (r = 0.20, p < 0.001, 300 train clips).
The downloaded model is a pickled timm model; it is loaded with
`weights_only=True`, allowing only its `torch.nn` / timm layer classes, and
copied into `tf_efficientnet_b0`, which reproduces its outputs exactly.

**Data:** `mosi_data.pkl` (MulT release), plus `Raw/` and `label.csv` from the
CMU-MOSI MMSA release. Sample IDs in the pickle match `Raw/<video_id>/<clip_id>.mp4`.

**1. Extract frame maps** (about 12 minutes on an RTX 3050, 4.4 GB):

```bash
cd MPLMM
python -m pip install hsemotion timm opencv-python
python tools/extract_mosi_spatial.py --backbone hsemotion_head --margin 1.2 \
  --data_path C:/mplmm_data/mosi_data.pkl \
  --raw_dir C:/mplmm_data/CMU-MOSI/Raw \
  --out_dir C:/mplmm_data/mosi_hsehead_t16
```

A face was found in 96–98% of frames per split. Frames without a detection
use the clip's median face box; 3 train clips had no face at all and used a
centre crop. `--backbone resnet18` gives the earlier ImageNet ResNet-18 maps.

**2. Fine-tune** with the baseline MOSI settings, choosing the checkpoint by
average validation accuracy over the six missing cases:

```bash
python main.py \
  --pretrained_model pretrained/mosei.pt \
  --dataset mosi \
  --data_path C:/mplmm_data/mosi_data.pkl \
  --spatial_feats C:/mplmm_data/mosi_hsehead_t16 \
  --drop_rate 0.7 --num_epochs 30 --seed 1111 --shuffle_train \
  --selection_metric avg_acc \
  --name results/sweep/mosi_hsehead_selacc_seed_1111.pt
```

`--spatial_pool` averages each map to 1×1 before the convolution. This is the
no-spatial control: same pipeline and features, without the spatial layout.

**3. Evaluate** the six fixed missing-modality cases (same protocol as the notebook):

```bash
python eval_missing.py \
  --checkpoint results/sweep/mosi_hsehead_selacc_seed_1111.pt \
  --data_path C:/mplmm_data/mosi_data.pkl \
  --spatial_feats C:/mplmm_data/mosi_hsehead_t16 \
  --split test
```

### Results (MOSI test, binary ACC / F1)

Spatial models: mean ± std over seeds 1111, 2222, 3333, 4444 and 5555, trained on an RTX 3050.
Baseline: the MOSI row of the main table (one seed, trained on a different machine).

| Model | {a} | {v} | {t} | {a,v} | {a,t} | {v,t} | **Avg.** |
|---|---|---|---|---|---|---|---|
| MPLMM baseline (FACET) | 47.23 / 36.99 | 48.69 / 40.30 | 75.07 / 75.08 | 50.87 / 45.63 | 75.95 / 75.90 | 75.51 / 75.49 | **62.22 / 58.23** |
| + 2D spatial, pooled (control) | 47.49 / 35.00 | 54.37 / 52.19 | 64.17 / 60.82 | 54.31 / 51.87 | 64.69 / 61.19 | 60.44 / 58.91 | **57.58 / 53.33** (± 5.5 / 9.5) |
| **+ 2D spatial** | 50.64 / 42.88 | 52.10 / 51.54 | 73.38 / 73.38 | 53.38 / 52.92 | 72.68 / 72.67 | 70.50 / 70.42 | **62.11 / 60.63** (± 1.4 / 1.7) |

- **Spatial vs pooled control (same features and protocol):** keeping the 7×7
  layout gives +4.5 ACC and +7.3 F1 on test, and +4.4 / +6.9 on validation. It
  is also much more stable across seeds. The pooled model collapses on the text
  cases for some seeds.
- **Spatial vs baseline:** average ACC is the same (−0.1) and average F1 is
  +2.4. Every case without text improves: {v} F1 rose from 40.3 to 51.5, {a,v}
  from 45.6 to 52.9 and {a} from 37.0 to 42.9. The text cases are lower
  ({t} −1.7, {a,t} −3.3, {v,t} −5.0 ACC).
- Vision-only accuracy (52.1%) is still below always predicting "negative"
  (55.25% of test labels), but its F1 is now balanced across classes.
- The baseline is a single seed from another machine. Spatial seed 1111 alone
  scores 63.61 / 61.99.

### How the final configuration was chosen

Every choice was made on the **validation** mean (seeds 1111–3333, or 1111–5555
where marked); test was not used for selection.

| Variant | Valid Avg. ACC | Test Avg. ACC / F1 |
|---|---|---|
| ResNet-18 maps (first version) | 62.90 | 60.65 / 59.79 |
| ResNet-18, 16 conv channels | 65.84 | 59.58 / 56.40 |
| ResNet-18, dropout 0.3 | 64.24 | 58.65 / 56.74 |
| ResNet-18, output size 32 | 63.05 | 59.26 / 57.28 |
| HSEmotion 320×7×7 (before the head) | 60.36 | 54.08 / 50.81 |
| HSEmotion 1280×7×7 head maps (5 seeds) | 65.33 | 60.19 / 58.02 (± 4.4 / 5.9) |
| HSEmotion head, 16 conv channels | 64.78 | 60.71 / 59.00 |
| **HSEmotion head + accuracy-based selection (5 seeds)** | **66.55** | **62.11 / 60.63** (± 1.4 / 1.7) |
| HSEmotion head + accuracy-based selection, lr 5e-4 (5 seeds) | 63.76 | 55.89 / 52.11 |

- **Accuracy-based selection** (`--selection_metric avg_acc`) mattered most for
  stability. The default keeps the epoch with the lowest validation loss, but
  that validation set re-samples random missing modalities every epoch, so it is
  noisy; one seed ended at 52.6 test accuracy. Selecting on the six fixed cases
  cut the seed spread by two thirds. This change also applies to the baseline,
  so part of the gain is protocol, not spatial; the pooled control above uses
  the same protocol.
- The first encoder had no normalization and was unstable (its output grew to
  about 28× the FACET scale; 55.2 / 52.8). Those results are in
  `MPLMM/results/v1_unnormalized/`.

Per-seed rows are in `MPLMM/results/spatial_results.csv` (ResNet-18) and
`MPLMM/results/sweep/*.csv` (everything else). Training logs are in
`MPLMM/results/logs/` and `MPLMM/results/sweep/logs/`. Checkpoints of the final
model and its control are in `MPLMM/results/sweep/`.

## Citation

```bibtex
@inproceedings{guo2024multimodal,
  title={Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition},
  author={Guo, Zirun and Jin, Tao and Zhao, Zhou},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1726--1736},
  year={2024}
}
```
