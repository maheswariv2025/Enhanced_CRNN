# Learning under Extreme Data Scarcity: An Enhanced Hybrid CRNN with Calibration and Test-Time Augmentation for Multi-Class Lung CT Classification

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Maheswari V and Parveen Sultana H**
> School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu 632014, India
> 📧 Correspondence: [hparveen.sultana@vit.ac.in](mailto:hparveen.sultana@vit.ac.in)

**An Enhanced Hybrid CRNN (EfficientNet-B0 + BiLSTM + Multi-Head Attention + Dual-Path Fusion) with temperature-scaled calibration and test-time augmentation for lung cancer subtype classification from CT images under extreme low-data constraints (~20 training samples).**

---

## Overview

This repository contains the complete implementation and reproducible experimental pipeline for our proposed **Enhanced Hybrid CRNN** architecture, which combines convolutional feature extraction with recurrent sequence modeling and attention mechanisms for lung CT classification. The model is designed to operate under a **5% coreset regime** (~20 training images selected via Farthest-Point Sampling), targeting realistic clinical scenarios where annotated medical data is scarce.

The framework incorporates **temperature-scaled calibration** for reliable confidence estimation and **test-time augmentation (TTA)** for robust inference. All models (proposed and baselines) are benchmarked under a **unified training protocol** (identical epoch budget, early stopping, and evaluation pipeline), with comprehensive ablation studies, calibration diagnostics, and statistical significance testing.

### Key Contributions

- **Dual-Path Fusion Architecture** — Combines a CNN classification path with a BiLSTM + Multi-Head Attention sequential path, fused via a learnable weight (α = 0.7), enabling the model to capture both spatial and sequential feature dependencies from CT feature maps.
- **Calibration and Confidence Reliability** — Applies post-hoc temperature scaling fitted on the validation set, reducing ECE from 0.1589 to 0.1190 across seeds, with reliability diagram analysis and adaptive ECE reporting for trustworthy clinical deployment.
- **Test-Time Augmentation** — Employs a 3-transform TTA ensemble (identity, horizontal flip, 5° rotation) that consistently improves accuracy and AUC over single-inference predictions.
- **Coreset Training with Farthest-Point Sampling** — Trains on only 5% of the data (~20 images) selected via class-stratified FPS in ResNet-18 embedding space, significantly outperforming random and stratified sampling baselines.
- **CLAHE-Based CT Preprocessing** — Applies Contrast-Limited Adaptive Histogram Equalization followed by morphological lung segmentation to enhance tissue contrast before feature extraction.
- **Rigorous Evaluation Protocol** — Multi-seed (3×) × multi-coreset (3×) = 9 independent runs per model with bootstrap confidence intervals, Wilcoxon signed-rank tests, Cohen's d effect sizes, and calibration diagnostics.

---

## Architecture

```
Input CT Image (224×224×3)
        │
   ┌────▼────┐
   │  CLAHE   │  Contrast enhancement + morphological lung segmentation
   │ Preproc  │
   └────┬────┘
        │
   ┌────▼─────────────┐
   │  EfficientNet-B0  │  Pretrained backbone (ImageNet)
   │  + SE Attention   │  Squeeze-and-Excitation channel attention
   └────┬──────────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────┐  ┌──────────────────────────────┐
│ CNN  │  │ 1×1 Conv Compress (→256)     │
│ Path │  │        ↓                     │
│ GAP  │  │ BiLSTM (128 hidden, bidir)   │
│  ↓   │  │        ↓                     │
│ FC   │  │ Multi-Head Attention (4 head)│
│  ↓   │  │        ↓                     │
│Logits│  │ Max+Mean Pool → FC → Logits  │
└──┬───┘  └──────────┬───────────────────┘
   │                 │
   └───────┬─────────┘
           ▼
   Dual-Path Fusion: α·RNN + (1-α)·CNN
           │
     4-class output
```

**Classes:** Adenocarcinoma · Large Cell Carcinoma · Squamous Cell Carcinoma · Normal

---

## Results

### Main Comparison (9 runs: 3 seeds × 3 coresets)

| Model | Accuracy | Macro-F1 | Macro-AUC | ECE |
|:------|:--------:|:--------:|:---------:|:---:|
| **Enhanced CRNN (Ours)** | **0.4523 ± 0.0085** | **0.4637 ± 0.0085** | **0.7302 ± 0.0073** | 0.1589 ± 0.0178 |
| Enhanced CRNN + TTA | 0.4557 ± 0.0085 | 0.4671 ± 0.0083 | 0.7357 ± 0.0069 | 0.1501 ± 0.0167 |
| Ensemble (Top-3) | 0.4105 ± 0.0075 | 0.3983 ± 0.0069 | 0.7428 ± 0.0072 | 0.1345 ± 0.0134 |
| ResNet-18 | 0.3842 ± 0.0089 | 0.3654 ± 0.0089 | 0.7073 ± 0.0073 | 0.1923 ± 0.0234 |
| ResNet-50 | 0.3567 ± 0.0098 | 0.3363 ± 0.0094 | 0.6801 ± 0.0082 | 0.2078 ± 0.0278 |
| DenseNet-121 | 0.3774 ± 0.0085 | 0.3585 ± 0.0088 | 0.6962 ± 0.0076 | 0.1956 ± 0.0256 |
| MobileNet-V3-Large | 0.3002 ± 0.0082 | 0.2713 ± 0.0081 | 0.5192 ± 0.0085 | 0.0823 ± 0.0189 |
| EfficientNet-B0 | 0.3447 ± 0.0074 | 0.3147 ± 0.0072 | 0.5510 ± 0.0082 | 0.0567 ± 0.0167 |
| ConvNeXt-Tiny | 0.3808 ± 0.0089 | 0.3623 ± 0.0084 | 0.7031 ± 0.0072 | 0.1878 ± 0.0223 |
| Random Baseline | 0.2542 | 0.2389 | 0.5000 | — |

The Enhanced CRNN achieves a **+17.7% relative improvement** in accuracy over the best baseline (ResNet-18) with statistical significance (p < 0.01, Wilcoxon signed-rank test) across all comparisons.

### Ablation Study

| Variant | Accuracy | Macro-F1 | Macro-AUC |
|:--------|:--------:|:--------:|:---------:|
| A: Backbone Only | 0.3509 ± 0.0042 | 0.3217 ± 0.0042 | 0.5545 ± 0.0089 |
| B: + BiLSTM | 0.4085 ± 0.0043 | 0.3962 ± 0.0042 | 0.6923 ± 0.0078 |
| C: + Attention | 0.3881 ± 0.0042 | 0.3728 ± 0.0043 | 0.6734 ± 0.0082 |
| **D: Full CRNN** | **0.4509 ± 0.0118** | **0.4623 ± 0.0112** | **0.7312 ± 0.0082** |

### Calibration & Test-Time Augmentation

| Configuration | Accuracy | Macro-F1 | ECE | Temperature |
|:--------------|:--------:|:--------:|:---:|:-----------:|
| Enhanced CRNN (base) | 0.4523 ± 0.0085 | 0.4637 ± 0.0085 | 0.1589 ± 0.0178 | — |
| + Temperature Scaling | — | — | **0.1190 ± 0.0037** | 1.16–1.35 |
| + Test-Time Augmentation | **0.4557 ± 0.0085** | **0.4671 ± 0.0083** | 0.1501 ± 0.0167 | — |

Temperature scaling reduces ECE by ~25% with stable temperature values (range 1.16–1.35) across all 9 seed configurations, demonstrating reliable post-hoc calibration for clinical confidence estimation.

### Per-Class Performance (Enhanced CRNN)

| Class | Precision | Recall | F1-Score | Support |
|:------|:---------:|:------:|:--------:|--------:|
| Adenocarcinoma | 0.5135 | 0.4865 | 0.4997 | 74 |
| Large Cell Carcinoma | 0.4312 | 0.3919 | 0.4107 | 74 |
| Squamous Cell Carcinoma | 0.4067 | 0.4459 | 0.4254 | 74 |
| Normal | 0.5534 | 0.5068 | 0.5291 | 73 |

### Coreset Strategy Comparison

| Strategy | Accuracy | Macro-F1 |
|:---------|:--------:|:--------:|
| Random | 0.4068 ± 0.0156 | 0.4156 ± 0.0178 |
| Stratified | 0.4271 ± 0.0112 | 0.4367 ± 0.0134 |
| **FPS (Ours)** | **0.4542 ± 0.0089** | **0.4645 ± 0.0098** |

### Fusion Weight Sensitivity

| α (RNN / CNN) | Accuracy | Macro-F1 | Macro-AUC |
|:--------------:|:--------:|:--------:|:---------:|
| 0.5 / 0.5 | 0.4305 | 0.4412 | 0.7134 |
| 0.6 / 0.4 | 0.4441 | 0.4545 | 0.7212 |
| **0.7 / 0.3** | **0.4576** | **0.4689** | **0.7356** |
| 0.8 / 0.2 | 0.4508 | 0.4612 | 0.7289 |
| 0.9 / 0.1 | 0.4373 | 0.4478 | 0.7178 |

### Statistical Significance

All comparisons between the Enhanced CRNN and baselines are statistically significant at p < 0.01 (Wilcoxon signed-rank test, 9 paired observations). Cohen's d effect sizes range from 7.89 to 22.45, indicating large practical differences. Full statistical test results are available in `tables/statistical_tests.csv`.

---

## Dataset

**Source:** [`dorsar/lung-cancer`](https://huggingface.co/datasets/dorsar/lung-cancer) on Hugging Face

| Split | Samples |
|:------|--------:|
| Train | 400 |
| Validation | 72 |
| Test | 295 |
| **Coreset (5%)** | **~20** |

The dataset contains CT scan images across four classes of lung tissue. All images are preprocessed through CLAHE contrast enhancement and morphological segmentation before training.

---

## Repository Structure

```
├── README.md
├── crnn_v2_experiment.py          # Main experiment script (all phases)
├── fig/                           # Generated figures
│   ├── figS1_learning_curves_all.png
│   ├── figS2_ablation.png
│   ├── fusion_sensitivity.png
│   ├── cm_*.png                   # Confusion matrices
│   ├── roc_*.png                  # ROC curves
│   ├── pr_*.png                   # Precision-Recall curves
│   ├── reliability_*.png          # Reliability diagrams
│   └── loss_*.png                 # Per-model learning curves
├── tables/                        # Generated CSV/JSON tables
│   ├── aggregated_results.csv
│   ├── statistical_tests.csv
│   ├── ablation_results.csv
│   ├── fusion_sensitivity.csv
│   ├── coreset_comparison.csv
│   ├── per_class_metrics.csv
│   ├── model_parameters.csv
│   ├── hyperparameters.json
│   ├── environment.json
│   └── coreset_indices_*.json     # Per-run coreset indices
└── ckpt/                          # Model checkpoints
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/lung-ct-hybrid-crnn.git
cd lung-ct-hybrid-crnn

# Create environment
conda create -n crnn python=3.10 -y
conda activate crnn

# Install dependencies
pip install torch torchvision
pip install numpy opencv-python-headless Pillow matplotlib scikit-learn scipy
pip install datasets huggingface_hub
```

### Requirements

| Package | Version |
|:--------|:--------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 2.0 |
| torchvision | ≥ 0.15 |
| numpy | ≥ 1.24 |
| scikit-learn | ≥ 1.2 |
| opencv-python | ≥ 4.7 |
| datasets | ≥ 2.14 |
| huggingface_hub | ≥ 0.17 |
| matplotlib | ≥ 3.7 |
| scipy | ≥ 1.10 |

---

## Usage

### Run Full Experiment

```bash
python crnn_v2_experiment.py
```

This runs all eight phases sequentially:

1. **Phase 1** — Multi-seed × multi-coreset training (9 runs × 7 models)
2. **Phase 2** — Aggregated results with bootstrap CIs and statistical tests
3. **Phase 3** — Ablation study (backbone-only → +BiLSTM → +Attention → full)
4. **Phase 4** — Fusion weight sensitivity sweep (α ∈ {0.5, 0.6, 0.7, 0.8, 0.9})
5. **Phase 5** — Coreset strategy comparison (Random vs. Stratified vs. FPS)
6. **Phase 6** — Figure generation (learning curves, confusion matrices, ROC/PR, reliability diagrams)
7. **Phase 7** — Per-class analysis and random baseline comparison
8. **Phase 8** — Calibration stability analysis across seeds

### Runtime Estimates

| Hardware | Estimated Time |
|:---------|:--------------:|
| CPU only | ~70–90 min |
| Single GPU (e.g., T4) | ~20–30 min |

### Google Colab

The script automatically detects Colab environments and uses `/content/drive/MyDrive/Maheswari/crnn_workspace_v2` as the output directory. Mount Google Drive before running:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Training Protocol

All models (proposed and baselines) are trained under a **unified protocol** to ensure fair comparison:

| Parameter | Baselines | Enhanced CRNN |
|:----------|:---------:|:-------------:|
| Max Epochs | 8 | 8 |
| Early Stopping Patience | 3 | 3 |
| Optimizer | AdamW | AdamW |
| Learning Rate | 3e-4 (flat) | 3e-4 head / 9e-5 backbone |
| Weight Decay | 1e-4 | 1e-4 |
| Scheduler | Cosine decay | Warmup (2 ep) + Cosine |
| Label Smoothing | 0.0 | 0.05 |
| Mixup | No | α = 0.1, p = 0.5 |
| Gradient Clipping | 1.0 | 1.0 |

---

## Preprocessing Pipeline

1. **Resize** to 224 × 224
2. **CLAHE** (clip limit = 2.0, tile = 8×8) for contrast enhancement
3. **Otsu thresholding** + morphological open/close for lung mask extraction
4. **Connected-component analysis** to isolate the largest lung region
5. **Background normalization** to mean background intensity
6. **ImageNet normalization** (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225])

Training augmentations include random resized crop, horizontal/vertical flips, rotation (±15°), color jitter, and affine translation.

---

## Evaluation Metrics

The framework reports a comprehensive set of metrics per model:

- **Classification:** Accuracy, Balanced Accuracy, Macro/Weighted F1, Macro Precision/Recall, Cohen's κ, Matthews Correlation Coefficient
- **Ranking:** Macro AUC-ROC (OvR), Macro AUPRC, Log Loss
- **Calibration:** ECE (15 equal-width bins), Adaptive ECE (15 equal-mass bins), Brier Score, Maximum Calibration Error, Temperature Scaling
- **Statistical:** Bootstrap 95% CI, Wilcoxon signed-rank test (or paired t-test for n < 5), Cohen's d effect size

---

## Reproducibility

Full reproducibility is ensured through:

- **Deterministic seeding** — All random seeds (Python, NumPy, PyTorch, CUDA) are set before each run
- **Logged coreset indices** — Exact training sample indices saved per run as JSON
- **Hyperparameter table** — Complete configuration exported to `hyperparameters.json`
- **Environment logging** — PyTorch version, hardware specs, and GPU memory recorded in `environment.json`
- **Multi-seed protocol** — Results aggregated over 9 independent runs (3 experiment seeds × 3 coreset seeds)

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{maheswari2025learning,
  title={Learning under Extreme Data Scarcity: An Enhanced Hybrid CRNN with Calibration and Test-Time Augmentation for Multi-Class Lung CT Classification},
  author={Maheswari, V and Parveen Sultana, H},
  journal={<Journal>},
  year={2025},
  institution={School of Computer Science and Engineering, Vellore Institute of Technology}
}
```

---

## Acknowledgements

- Dataset provided by [`dorsar/lung-cancer`](https://huggingface.co/datasets/dorsar/lung-cancer) on Hugging Face
- Backbone pretrained weights from [TorchVision](https://pytorch.org/vision/stable/models.html) (ImageNet-1K)
- School of Computer Science and Engineering, Vellore Institute of Technology, Vellore

---

## License

This project is released under the [MIT License](LICENSE).
