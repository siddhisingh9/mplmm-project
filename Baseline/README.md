# Baseline

Reproduction workspace for the MPLMM multimodal learning experiments, including
training code, diagnostics, and notebook-based Figure 4 and Figure 6 analysis.

The implementation is based on the official
[MPLMM](https://github.com/zrguo/MPLMM) code for:

> Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and
> Emotion Recognition

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

## Training

From the `MPLMM` directory, use the same Python interpreter as the notebook:

```bash
python main.py \
  --dataset mosi \
  --data_path dataset/mosi_data.pkl \
  --drop_rate 0.7 \
  --name results/mosi_stage2.pt \
  --num_epochs 30
```

For transfer learning, provide the pretrained MOSEI checkpoint with
`--pretrained_model pretrained/mosei.pt`.

## Figures

The notebook contains the deterministic evaluation and plotting cells for the
missing-modality experiments. Generated Figure 4 files are saved as:

```text
MPLMM/results/CMU-MOSI Performance During Training.png
MPLMM/results/MPLMM Performance Under Different Test-Time Missing Rates.png
MPLMM/results/table.png

```

Generated Figure 6 files are saved under `MPLMM/results/` when the corresponding
training runs complete.

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
