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
