<div align="center">

# SWIFTFormer

### Spatial-Wavelet Integrated Frequency Transformer for Robust Low-Light Image Enhancement

[Debashis Das](mailto:debashis_2221cs31@iitp.ac.in) · [Suman Kumar Maji](https://www.iitp.ac.in/~smaji/)

Department of Computer Science and Engineering, Indian Institute of Technology Patna, India

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-00629B?logo=ieee&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f)](LICENSE)
[![Visual Results](https://img.shields.io/badge/Visual%20Results-Google%20Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX)

<img src="figures/SWIFTFormer_Structure.png" width="95%">

</div>

---

## Overview

SWIFTFormer is a **dual-domain hierarchical Transformer** for low-light image enhancement. It decouples illumination correction from structural restoration by processing features jointly in the **spatial** and **wavelet-frequency** domains:

- **GICB / GEB** — a gradient-infused convolutional stem that injects multi-directional (h / v / d1 / d2) edge priors before the encoder.
- **SCSD-MHA** — Spatial-Channel Shifted Depthwise Multi-Head Attention using **L2-normalized cosine similarity**, providing intensity-scale invariance and stable attention under extreme, uneven illumination at O(d) projection cost.
- **SSF-FFN** — Spatial-Spectral Fusion Feed-Forward Network: a DWT (db4) splits features into a low-frequency **Illumination-Semantic Subspace** and a high-frequency **Structural-Detail Subspace**, each refined by dedicated convolutional cascades and recombined via IDWT.
- **SSAG** — a learnable Spectro-Spatial Attention Gate that adaptively blends the spatial and spectral streams pixel-wise: `G · X_spatial + (1 − G) · X_spectral`.

The result: **12.46 M parameters, 29.31 GMacs, 0.137 s inference** — with state-of-the-art PSNR / SSIM / LPIPS / NIQE across eight paired and unpaired benchmarks.

---

## News

| Date | Update |
| :-- | :-- |
| **Jul 2026** | Code, training configs, and evaluation scripts released |
| **Jul 2026** | Paper submitted to *IEEE Access* |
| *Coming soon* | Pre-trained model zoo and full visual results |

---

## Quantitative Results

<details open>
<summary><b>Paired benchmarks — PSNR↑ / SSIM↑ / LPIPS↓</b></summary>
<br>

| Method | LOL | LOLv2-Synthetic | LOLv2-Real | MIT-Adobe 5K | Avg. PSNR |
| :-- | :--: | :--: | :--: | :--: | :--: |
| Restormer | 22.36 / 0.853 / 0.154 | 24.37 / 0.901 / 0.085 | 21.80 / 0.867 / 0.118 | 23.59 / 0.912 / 0.072 | 23.53 |
| LLFormer | 23.64 / 0.859 / 0.133 | 24.82 / 0.909 / 0.076 | 22.10 / 0.870 / 0.115 | 23.80 / 0.914 / 0.071 | 23.59 |
| RetinexFormer | 24.01 / 0.861 / 0.120 | 25.76 / 0.917 / 0.072 | 22.63 / 0.871 / 0.115 | 24.42 / 0.919 / 0.068 | 24.21 |
| SNR-Aware | 24.21 / 0.865 / 0.119 | 25.81 / 0.921 / 0.068 | 22.69 / 0.870 / 0.112 | 24.53 / 0.922 / 0.065 | 24.31 |
| LIEDNet | 24.61 / 0.861 / 0.126 | 26.03 / 0.925 / 0.055 | 22.36 / 0.866 / 0.116 | 24.20 / 0.918 / 0.058 | 24.30 |
| **SWIFTFormer (Ours)** | **24.74 / 0.872 / 0.114** | **26.32 / 0.937 / 0.039** | **22.71 / 0.877 / 0.110** | **25.47 / 0.928 / 0.044** | **24.80** |

</details>

<details>
<summary><b>Unpaired real-world benchmarks — NIQE↓</b></summary>
<br>

| Method | DICM | LIME | MEF | NPE |
| :-- | :--: | :--: | :--: | :--: |
| Restormer | 3.961 | 4.103 | 4.680 | 4.258 |
| CWNet | 3.812 | 3.734 | 3.796 | 3.792 |
| LIEDNet | 3.725 | 3.652 | 3.728 | 3.671 |
| **SWIFTFormer (Ours)** | **3.655** | **3.612** | **3.710** | **3.625** |

</details>

<details>
<summary><b>Computational efficiency — 256 × 256 patch</b></summary>
<br>

| Method | Params (M)↓ | MACs (G)↓ | Time (s)↓ | PSNR (dB)↑ | SSIM↑ |
| :-- | :--: | :--: | :--: | :--: | :--: |
| HWMNet | 66.56 | 98.00 | 0.52 | 24.24 | 0.853 |
| Restormer | 26.10 | 72.12 | 0.61 | 24.36 | 0.852 |
| LLFormer | 24.55 | – | 0.82 | 23.64 | 0.859 |
| LIEDNet | 17.79 | 83.95 | 0.55 | 24.61 | 0.861 |
| **SWIFTFormer (Ours)** | **12.46** | **29.31** | **0.137** | **24.74** | **0.872** |

</details>

---

## Visual Results

Enhanced outputs of SWIFTFormer on all benchmarks are available for download. Each archive contains the full set of enhanced images used in the paper, organized per dataset.

| Benchmark | Type | Enhanced Results |
| :-- | :-- | :--: |
| LOL | Paired | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| LOLv2-Synthetic | Paired | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| LOLv2-Real | Paired | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| MIT-Adobe FiveK | Paired | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| DICM | Unpaired | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| LIME | Unpaired | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| MEF | Unpaired | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| NPE | Unpaired | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| **All results (single archive)** | — | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |

> To reproduce any number reported above, download the corresponding folder and run
> `python evaluation.py --result_dir <downloaded_folder> --gt_dir <dataset>/test/high`.

---

## Installation

```bash
git clone https://github.com/<user>/SWIFTFormer.git
cd SWIFTFormer

conda create -n swiftformer python=3.10 -y
conda activate swiftformer

pip install -r requirements.txt
```

> **Note:** `pytorch-wavelets` provides the four-tap Daubechies (db4) DWT used in the paper. If unavailable, the model transparently falls back to an exact built-in Haar transform.

---

## Dataset Preparation

| Dataset | Type | Source |
| :-- | :-- | :-- |
| LOL | Paired | [Project page](https://daooshee.github.io/BMVC2018website/) |
| LOL-v2 (Real / Synthetic) | Paired | [GitHub](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) |
| MIT-Adobe FiveK | Paired | [Project page](https://data.csail.mit.edu/graphics/fivek/) |
| DICM / LIME / MEF / NPE | Unpaired | Standard LLIE evaluation sets |

Arrange the data as:

```
datasets/
├── LOL/
│   ├── train/{low, high}/
│   └── test/{low, high}/
├── LOLv2/
│   ├── Real_captured/{train, test}/{low, high}/
│   └── Synthetic/{train, test}/{low, high}/
├── MIT-Adobe-FiveK/{train, test}/{low, high}/
├── DICM/          # unpaired: images only
├── LIME/
├── MEF/
└── NPE/
```

---

## Training

Training follows the paper's protocol: Adam (β₁ = 0.9, β₂ = 0.999), ~2.5×10⁵ iterations, learning rate 2e-4 → 1e-6 with cosine annealing, 128 × 128 random crops, batch size 8, rotation + flip augmentation, Charbonnier loss.

```bash
# LOL
python train.py --yml_path configs/LOL/train/training_LOL.yaml

# LOLv2-Real
python train.py --yml_path configs/LOLv2-Real/train/training_LOLv2_real.yaml

# LOLv2-Synthetic
python train.py --yml_path configs/LOLv2-Synthetic/train/training_LOLv2_synthetic.yaml

# MIT-Adobe FiveK
python train.py --yml_path configs/MIT-Adobe-FiveK/train/training_MIT_5K.yaml
```

---

## Testing

**Paired / synthetic benchmarks** — reports PSNR, SSIM, and LPIPS:

```bash
python test_synthetic.py \
    --input_dir ./datasets/LOL/test \
    --result_dir ./results/LOL \
    --weights ./checkpoints/SWIFTFormer_LOL.pth
```

Applicable to LOL, LOLv2-Synthetic, LOLv2-Real, and MIT-Adobe FiveK — point `--input_dir` at the corresponding `test/` folder.

**Real-world / unpaired benchmarks** — reports NIQE (no ground truth needed):

```bash
python test_real.py \
    --input_dir ./datasets/DICM \
    --result_dir ./results/DICM \
    --weights ./checkpoints/SWIFTFormer_LOL.pth
```

Applicable to DICM, LIME, MEF, and NPE. Add `--max_size 1600` to bound memory on very large images.

**Utilities:**

```bash
python evaluation.py --result_dir ./results/LOL --gt_dir ./datasets/LOL/test/high   # metrics on saved results
python demo.py --input_dir ./demo/input --result_dir ./demo/output --weights ./checkpoints/SWIFTFormer_LOL.pth
python scripts/compute_complexity.py                                               # params / MACs / runtime
```

---

## Model Zoo

| Trained on | Test set | PSNR (dB) | SSIM | Weights |
| :-- | :-- | :--: | :--: | :--: |
| LOL | LOL | 24.74 | 0.872 | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| LOLv2-Synthetic | LOLv2-Synthetic | 26.32 | 0.937 | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| LOLv2-Real | LOLv2-Real | 22.71 | 0.877 | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |
| MIT-Adobe FiveK | MIT-Adobe FiveK | 25.47 | 0.928 | [Google Drive](https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXX) |

---

## Repository Structure

```
SWIFTFormer/
├── model/SWIFTFormer.py          # GICB · SCSD-MHA · SSF-FFN · SSAG · H-ENC/H-DEC
├── datasets/dataset.py           # paired train/val + unpaired real-world loaders
├── utils/                        # IO, checkpointing, Charbonnier loss
├── configs/                      # YAML training configs (4 benchmarks)
├── train.py                      # training pipeline
├── test_synthetic.py             # paired evaluation  (PSNR / SSIM / LPIPS)
├── test_real.py                  # unpaired evaluation (NIQE)
├── evaluation.py                 # metrics on saved result folders
├── demo.py                       # quick folder inference
└── scripts/compute_complexity.py # params / MACs / runtime profiling
```

---

## Citation

```bibtex
@article{das2026swiftformer,
  title   = {SWIFTFormer: Spatial-Wavelet Integrated Frequency Transformer
             for Robust Low-Light Image Enhancement},
  author  = {Das, Debashis and Maji, Suman Kumar},
  journal = {IEEE Access},
  year    = {2026}
}
```

## Acknowledgements

This repository follows engineering conventions from [Restormer](https://github.com/swz30/Restormer), [LLFormer](https://github.com/TaoWangzj/LLFormer), and [BasicSR](https://github.com/XPixelGroup/BasicSR). We thank the authors for open-sourcing their work.

## Contact

For questions, please open an issue or contact **Debashis Das** — `debashis_2221cs31@iitp.ac.in`.
