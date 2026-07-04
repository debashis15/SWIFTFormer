<div align="center">
⚡ SWIFTFormer ⚡
Spatial-Wavelet Integrated Frequency Transformer for Robust Low-Light Image Enhancement
Debashis Das<sup>1</sup>, Suman Kumar Maji<sup>1</sup>
<sup>1</sup> Department of Computer Science and Engineering, Indian Institute of Technology Patna, India
[![paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue.svg)]()
![python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![pytorch](https://img.shields.io/badge/PyTorch-1.11%2B-ee4c2c.svg)
![license](https://img.shields.io/badge/License-MIT-green.svg)
[![params](https://img.shields.io/badge/Params-12.46M-orange.svg)]()
[![GitHub Stars](https://img.shields.io/github/stars/swiftformer/SWIFTFormer?style=social)]()
A dual-domain Transformer that decouples illumination from structure — spatial attention for global context, wavelet spectra for fine detail — and fuses them with a learnable Spectro-Spatial Attention Gate.
</div>
---
> **Abstract:** *This paper presents SWIFTFormer (Spatial-Wavelet Integrated Frequency Transformer), a dual-domain architecture for robust image enhancement under low-light conditions. Unlike conventional spatial-only frameworks, SWIFTFormer decouples feature extraction through two complementary modules: SCSD-MHA (Spatial-Channel Shifted Depthwise Multi-Head Attention) and SSF-FFN (Spatial-Spectral Fusion Feed-Forward Network). SCSD-MHA captures long-range dependencies via cosine-similarity-based attention, while SSF-FFN leverages discrete wavelet transform (DWT) to process multi-scale frequency components. This dual-pathway design enables independent optimization of illumination-related low-frequency information and structure-preserving high-frequency details. The Spectro-Spatial Attention Gate (SSAG) adaptively fuses spatial and spectral features through learnable channel-wise attention, dynamically modulating illumination and structural components based on local content. Extensive experiments on LOL, LOL-v2, LIME, MEF, NPE, and DICM benchmarks demonstrate state-of-the-art results across PSNR, SSIM, LPIPS and NIQE metrics.*
---
🔥 News
[2026-07] 🚀 Code, configs, and evaluation scripts are released!
[2026-07] 📄 SWIFTFormer paper submitted to IEEE Access.
[Coming soon] 🎯 Pre-trained model zoo (LOL / LOLv2-Real / LOLv2-Synthetic / MIT-Adobe FiveK).
✨ Highlights
🌗 Dual-domain decoupling — the DWT splits features into an Illumination-Semantic Subspace (LL) and a Structural-Detail Subspace (LH/HL/HH), so brightness is corrected without saturating textures.
📐 Cosine-similarity attention (SCSD-MHA) — L2-normalized affinity is invariant to intensity scaling, giving stable attention under extreme, uneven illumination.
🌀 Wavelet feed-forward (SSF-FFN) — dilated-conv cascade (π = {3, 2, 1}) on low frequencies + depthwise hierarchy (ω = {5, 3, 2}) on high frequencies, recombined by IDWT.
🚪 Spectro-Spatial Attention Gate (SSAG) — a pixel-wise sigmoid gate `G` blends spatial and spectral streams: `G·X_spa + (1−G)·X_spec`.
🧭 Gradient-Infused stem (GICB + GEB) — four-direction Sobel priors (h/v/d1/d2) inject edge awareness before the hierarchy.
🪶 Lightweight — 12.46 M parameters and 29.31 GMacs, with 0.137 s inference: ~75% fewer parameters and ~77% fewer MACs than the vanilla Transformer baseline.
🔌 Plug-and-play SSF-FFN — dropping our FFN into Restormer / LLFormer / WalMaFa yields +1.36 / +0.66 / +0.32 dB without fine-tuning.
🏗️ Network Architecture
<div align="center">
<img src="figures/SWIFTFormer_Structure.png" width="95%">
<p><i>Overall architecture of SWIFTFormer: a 4-scale hierarchical encoder–decoder (H-ENC / H-DEC) of Transformer blocks, each composed of RMSN → SCSD-MHA → RMSN → SSF-FFN, with a Gradient-Infused Convolutional Block (GICB) stem and a refinement stage.</i></p>
</div>
📊 Results
<details open>
<summary><b>Paired benchmarks (PSNR↑ / SSIM↑ / LPIPS↓)</b></summary>
Method	LOL	LOLv2-Syn	LOLv2-Real	MIT-Adobe 5K	Avg. PSNR
Restormer	22.36 / 0.853 / 0.154	24.37 / 0.901 / 0.085	21.80 / 0.867 / 0.118	23.59 / 0.912 / 0.072	23.53
LLFormer	23.64 / 0.859 / 0.133	24.82 / 0.909 / 0.076	22.10 / 0.870 / 0.115	23.80 / 0.914 / 0.071	23.59
RetinexFormer	24.01 / 0.861 / 0.120	25.76 / 0.917 / 0.072	22.63 / 0.871 / 0.115	24.42 / 0.919 / 0.068	24.21
SNR-Aware	24.21 / 0.865 / 0.119	25.81 / 0.921 / 0.068	22.69 / 0.870 / 0.112	24.53 / 0.922 / 0.065	24.31
LIEDNet	24.61 / 0.861 / 0.126	26.03 / 0.925 / 0.055	22.36 / 0.866 / 0.116	24.20 / 0.918 / 0.058	24.30
SWIFTFormer (Ours)	24.74 / 0.872 / 0.114	26.32 / 0.937 / 0.039	22.71 / 0.877 / 0.110	25.47 / 0.928 / 0.044	24.80 🏆
</details>
<details>
<summary><b>Unpaired real-world benchmarks (NIQE↓)</b></summary>
Method	DICM	LIME	MEF	NPE
Restormer	3.961	4.103	4.680	4.258
CWNet	3.812	3.734	3.796	3.792
LIEDNet	3.725	3.652	3.728	3.671
SWIFTFormer (Ours)	3.655	3.612	3.710	3.625 🏆
</details>
<details>
<summary><b>Efficiency (256×256 patch)</b></summary>
Method	Params (M)↓	MACs (G)↓	Time (s)↓	PSNR (dB)↑	SSIM↑
HWMNet	66.56	98.00	0.52	24.24	0.853
Restormer	26.10	72.12	0.61	24.36	0.852
LLFormer	24.55	–	0.82	23.64	0.859
LIEDNet	17.79	83.95	0.55	24.61	0.861
SWIFTFormer (Ours)	12.46	29.31	0.137	24.74	0.872 🏆
</details>
🛠️ Installation
```bash
git clone https://github.com/<user>/SWIFTFormer.git
cd SWIFTFormer

conda create -n swiftformer python=3.10 -y
conda activate swiftformer

pip install -r requirements.txt
```
> `pytorch-wavelets` provides the four-tap Daubechies (db4) DWT used in the paper. If it is unavailable, the model transparently falls back to an exact built-in Haar transform.
📁 Dataset Preparation
Download the benchmarks and arrange them as:
```
datasets/
├── LOL/                       # https://daooshee.github.io/BMVC2018website/
│   ├── train/{low, high}/
│   └── test/{low, high}/
├── LOLv2/
│   ├── Real_captured/{train, test}/{low, high}/
│   └── Synthetic/{train, test}/{low, high}/
├── MIT-Adobe-FiveK/{train, test}/{low, high}/
├── DICM/                      # unpaired: images only
├── LIME/
├── MEF/
└── NPE/
```
🚀 Training
All hyper-parameters follow the paper (Adam β = (0.9, 0.999), 2.5×10⁵ iterations, lr 2e-4 → 1e-6 cosine annealing, 128×128 crops, batch 8, rotation + flip augmentation, Charbonnier loss):
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
🔎 Testing
1️⃣ Synthetic / paired benchmarks (PSNR · SSIM · LPIPS)
```bash
python test_synthetic.py \
    --input_dir ./datasets/LOL/test \
    --result_dir ./results/LOL \
    --weights ./checkpoints/SWIFTFormer_LOL.pth
```
Works for LOL, LOLv2-Synthetic, LOLv2-Real, and MIT-Adobe FiveK — just point `--input_dir` at the corresponding `test/` folder.
2️⃣ Real-world / unpaired benchmarks (NIQE)
```bash
python test_real.py \
    --input_dir ./datasets/DICM \
    --result_dir ./results/DICM \
    --weights ./checkpoints/SWIFTFormer_LOL.pth
```
Works for DICM, LIME, MEF, and NPE (no ground truth required). Add `--max_size 1600` to bound memory on very large images.
3️⃣ Evaluate saved results / quick demo
```bash
python evaluation.py --result_dir ./results/LOL --gt_dir ./datasets/LOL/test/high
python demo.py --input_dir ./demo/input --result_dir ./demo/output --weights ./checkpoints/SWIFTFormer_LOL.pth
python scripts/compute_complexity.py     # params / MACs / runtime
```
🧩 Model Zoo
Trained on	Test set	PSNR	SSIM	Weights
LOL	LOL	24.74	0.872	coming soon
LOLv2-Synthetic	LOLv2-Syn	26.32	0.937	coming soon
LOLv2-Real	LOLv2-Real	22.71	0.877	coming soon
MIT-Adobe FiveK	FiveK	25.47	0.928	coming soon
📂 Repository Structure
```
SWIFTFormer/
├── model/SWIFTFormer.py        # GICB · SCSD-MHA · SSF-FFN · SSAG · H-ENC/H-DEC
├── datasets/dataset.py         # paired train/val + unpaired real-world loaders
├── utils/                      # IO, checkpointing, Charbonnier loss
├── configs/                    # YAML training configs (4 benchmarks)
├── train.py                    # training pipeline
├── test_synthetic.py           # paired evaluation  (PSNR / SSIM / LPIPS)
├── test_real.py                # unpaired evaluation (NIQE)
├── evaluation.py               # metrics on saved result folders
├── demo.py                     # quick single-folder inference
└── scripts/compute_complexity.py
```
📜 Citation
If you find SWIFTFormer useful in your research, please consider citing:
```bibtex
@article{das2026swiftformer,
  title   = {SWIFTFormer: Spatial-Wavelet Integrated Frequency Transformer
             for Robust Low-Light Image Enhancement},
  author  = {Das, Debashis and Maji, Suman Kumar},
  journal = {IEEE Access},
  year    = {2026}
}
```
🙏 Acknowledgements
This repository borrows engineering conventions from the excellent
Restormer, LLFormer, and BasicSR codebases — thanks to the authors for open-sourcing their work.
📧 Contact
For questions, please open an issue or contact Debashis Das — `debashis_2221cs31@iitp.ac.in`.
<div align="center">
⭐ If this project helps you, please give it a star! ⭐
</div>
