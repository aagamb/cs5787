# A Comparative Study on Deep Learning Architectures for Protein Localization

This repository contains the complete implementation for a systematic ablation study comparing deep learning architectures, loss functions, and augmentation strategies for protein localization classification using the Human Protein Atlas (HPA) Single-Cell Classification dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Authors](#authors)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Experimental Setup](#experimental-setup)
- [Key Results](#key-results)
- [Usage Instructions](#usage-instructions)
- [Key Findings](#key-findings)
- [References](#references)

## Overview

This project presents a comprehensive comparison of deep learning architectures for protein localization classification. We conducted a systematic ablation study across:

- **6 Backbone Architectures**: NFNet-L0, ResNet50, InceptionV3, EfficientNet-B0, EfficientNet-B3, EfficientNet-B5
- **2 Loss Functions**: Binary Cross-Entropy (BCE) and Focal Loss
- **2 Augmentation Strategies**: Standard geometric augmentations and CutMix

The goal is to provide evidence-based guidance for designing practical deep learning pipelines for protein localization tasks, especially for laboratories with limited computational resources.

## Authors

- **Aagam Bakliwal** (ab3239) - Cornell University
- **Shrey Verma** (sv528) - Cornell University  
- **Anushka Naik** (an645) - Cornell University

## Problem Statement

While convolutional neural networks (CNNs) are widely used for microscopy-based protein localization, it remains unclear which specific combinations of architectures, loss functions, and augmentation strategies perform best for the HPA single-cell classification problem. This work addresses this gap by developing a unified benchmarking framework that compares key pipeline components under consistent conditions.

## Dataset

### Human Protein Atlas (HPA) Single-Cell Classification Dataset

The dataset contains fluorescence microscopy images capturing protein distributions within individual human cells. Each image contains multiple cells with distinct expression patterns.

- **Task**: Multi-label classification - assign one or more subcellular localization labels to each cell
- **Number of Classes**: 19 subcellular localization classes
- **Image Format**: Multi-channel fluorescence microscopy images (using green channel)
- **Image Size**: Resized to 256×256 pixels
- **Input Format**: Single channel (green channel) converted to RGB-compatible format

### Class Labels

```
0:  Nucleoplasm
1:  Nuclear membrane
2:  Nucleoli
3:  Nucleoli fibrillar center
4:  Nuclear speckles
5:  Nuclear bodies
6:  Endoplasmic reticulum
7:  Golgi apparatus
8:  Intermediate filaments
9:  Actin filaments
10: Microtubules
11: Mitotic spindle
12: Centrosome
13: Plasma membrane
14: Mitochondria
15: Aggresome
16: Cytosol
17: Vesicles and punctate cytosolic patterns
18: Negative
```

## Methodology

### Model Architectures

All models use ImageNet-pretrained backbones with a custom classifier head:

1. **NFNet-L0** (`nfnet_f1`): Normalization-Free Network with gradient clipping
2. **ResNet50** (`resnet50`): Residual network with skip connections
3. **InceptionV3** (`inception_v3`): Multi-scale feature extraction
4. **EfficientNet-B0/B3/B5** (`efficientnet_b0/b3/b5`): Compound scaling strategy

**Classifier Head**: All models use a two-layer MLP (backbone features → 512 → 19) with ReLU activation.

### Loss Functions

1. **Binary Cross-Entropy with Logits (BCE)**: Standard multi-label loss
   ```python
   nn.BCEWithLogitsLoss()
   ```

2. **Focal Loss**: Addresses class imbalance by focusing on hard examples
   - Alpha: 0.25
   - Gamma: 2.0
   - Reduction: mean

### Augmentation Strategies

1. **Standard Augmentation**:
   - Random horizontal flip
   - Random vertical flip
   - Random rotation (±20 degrees)
   - Random affine translation (±10%)
   - ImageNet normalization

2. **CutMix Augmentation**:
   - Beta distribution parameter: 1.0
   - Application probability: 0.5
   - Mixes patches between images during training

### Training Configuration

- **Batch Size**: 32
- **Epochs**: 20
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Input Resolution**: 256×256
- **Device**: CUDA (if available)

### Evaluation Metrics

- **Macro F1 Score**: Primary metric for multi-label classification
- **Mean Average Precision (mAP)**: Measures ranking quality across all classes
- **Multi-label Confusion Matrix**: Per-class performance analysis

## Repository Structure

```
cs5787/
├── readme.md
├── EfNet B0/
│   ├── ef-standard-bce.ipynb
│   ├── ef-standard-focal.ipynb
│   ├── ef-cutmix-bce.ipynb
│   └── ef-cutmix-focal.ipynb
├── EfNet B3/
│   ├── ef-standard-bce.ipynb
│   ├── ef-standard-focal.ipynb
│   ├── ef-cutmix-bce.ipynb
│   └── ef-cutmix-focal.ipynb
├── EfNet B5/
│   ├── ef-standard-bce.ipynb
│   ├── ef-standard-focal.ipynb
│   ├── ef-cutmix-bce.ipynb
│   └── ef-cutmix-focal.ipynb
├── InNet/
│   ├── in-standard-bce.ipynb
│   ├── in-standard-focal.ipynb
│   ├── in-cutmix-bce.ipynb
│   └── in-cutmix-focal.ipynb
├── NFNet/
│   ├── ml-standard-bce.ipynb
│   ├── ml-standard-focal.ipynb
│   ├── ml-cutmix-bce.ipynb
│   └── ml-cutmix-focal.ipynb
└── ResNet/
    ├── rn-standard-bce.ipynb
    ├── rn-standard-focal.ipynb
    ├── rn-cutmix-bce.ipynb
    └── rn-cutmix-focal.ipynb
```

### Naming Convention

- **Architecture Prefixes**:
  - `ef-`: EfficientNet
  - `in-`: Inception
  - `ml-`: NFNet (Multi-Label)
  - `rn-`: ResNet

- **File Naming Pattern**:
  - `{arch}-standard-bce.ipynb`: Standard augmentation + BCE loss
  - `{arch}-standard-focal.ipynb`: Standard augmentation + Focal loss
  - `{arch}-cutmix-bce.ipynb`: CutMix augmentation + BCE loss
  - `{arch}-cutmix-focal.ipynb`: CutMix augmentation + Focal loss

## Experimental Setup

### Data Processing

1. **Dataset Class**: `HPADataset` loads single-channel images, resizes to 256×256, and applies transformations
2. **Label Encoding**: Multi-label binarization using `MultiLabelBinarizer` from scikit-learn
3. **Train/Validation Split**: Random split from training data
4. **Image Preprocessing**: OpenCV for loading, PIL/torchvision for transforms

### Code Components

Each notebook contains:

1. **Imports and Configuration**: All necessary libraries and hyperparameters
2. **Dataset Class**: Custom PyTorch Dataset for HPA images
3. **Model Architecture**: Custom wrapper class for each backbone
4. **Loss Function**: Either BCE or Focal Loss implementation
5. **Augmentation**: Standard transforms or CutMix dataset wrapper
6. **Training Loop**: Full training with validation evaluation
7. **Evaluation Functions**: F1 score, mAP calculation
8. **Test Functions**: Inference on test set

### Dependencies

Key libraries used:
- `torch`, `torchvision`: Deep learning framework
- `timm`: Pretrained model library
- `numpy`, `pandas`: Data manipulation
- `opencv-python`: Image loading
- `scikit-learn`: Metrics and label encoding
- `matplotlib`, `seaborn`: Visualization
- `tqdm`: Progress bars

## Key Results

Based on the comprehensive ablation study:

### Top Performing Configurations

1. **NFNet-L0 with Standard Augmentation**: Best overall performance
   - Strong performance across metrics
   - Stable training dynamics
   - Normalization-free design handles heterogeneous fluorescence intensities

2. **EfficientNet-B3 with Standard Augmentation**: Best balance
   - Strong accuracy with computational efficiency
   - Good alternative for resource-limited environments
   - Compound scaling captures fine-grained spatial details

3. **EfficientNet-B5**: Higher capacity variant
   - Strong performance but higher computational cost
   - Useful when accuracy is prioritized over efficiency

### Key Findings

1. **Standard Augmentation Outperforms CutMix**:
   - Protein localization depends on coherent spatial structure
   - CutMix introduces biologically implausible configurations
   - Larger models (NFNet-L0, EfficientNet-B5) especially sensitive to this

2. **Focal Loss Shows Modest Improvements**:
   - Small but meaningful improvements with standard augmentation
   - Benefits specific architectures (NFNet-L0, EfficientNet-B3, ResNet50)
   - Inconsistent gains and increased training time
   - Negative interaction with CutMix

3. **Architectural Differences Matter**:
   - Modern architectures (NFNet, EfficientNet) outperform traditional ones
   - Inductive biases and scaling strategies matter more than just depth
   - ResNet50 and InceptionV3 lagged behind consistently

4. **Error Analysis**:
   - Most misclassifications in rare classes
   - Difficulty with overlapping/diffuse patterns (e.g., nucleus vs nucleoplasm)
   - Limited by 256×256 input resolution

## Usage Instructions

### Prerequisites

1. **Dataset**: Download HPA Single-Cell Classification dataset
   - Place in `/kaggle/input/hpa-single-cell-image-classification/` (Kaggle format)
   - Or modify `base` path in notebooks to point to your dataset location

2. **Required Directories**:
   ```
   base/
   ├── train.csv          # Training labels
   ├── train/             # Training images (ID_green.png format)
   ├── test/              # Test images (optional)
   └── sample_submission.csv
   ```

3. **Environment Setup**:
   ```bash
   pip install torch torchvision
   pip install timm
   pip install numpy pandas opencv-python
   pip install scikit-learn matplotlib seaborn tqdm
   ```

### Running Experiments

1. **Choose Configuration**: Navigate to the desired architecture folder
2. **Select Notebook**: Choose augmentation + loss combination
3. **Update Paths**: Modify `base` variable to point to your dataset
4. **Run Cells**: Execute all cells sequentially

### Example: Training EfficientNet-B3 with Standard Augmentation and BCE

```python
# Navigate to EfNet B3/ef-standard-bce.ipynb
# Update base path:
base = "/path/to/your/dataset/"

# Model initialization:
model = EfficientNet(CLASS, model_name='efficientnet_b3')
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop runs automatically
```

### Customization

- **Hyperparameters**: Modify `BATCH_SIZE`, `EPOCHS`, `LR` in the configuration cell
- **Model Architecture**: Change `model_name` parameter (e.g., `'efficientnet_b0'` → `'efficientnet_b3'`)
- **Loss Function**: Switch between `nn.BCEWithLogitsLoss()` and `FocalLoss()`
- **Augmentation**: Toggle between `img_tfms` and `CutMixDataset` wrapper

## Key Findings

### Recommendations

1. **For Maximum Accuracy**:
   - Use NFNet-L0 with standard augmentation
   - Consider EfficientNet-B5 if computational budget allows

2. **For Balanced Performance**:
   - Use EfficientNet-B3 with standard augmentation
   - Optimal trade-off between accuracy and efficiency

3. **For Limited Resources**:
   - Use EfficientNet-B0 with standard augmentation
   - Still competitive while being most efficient

4. **Loss Function Selection**:
   - Focal Loss: Use when rare class detection is critical
   - BCE: Use for standard multi-label classification (simpler, faster)

5. **Augmentation Strategy**:
   - Always use standard geometric augmentations
   - Avoid CutMix for protein localization tasks

### Limitations and Future Work

1. **Interpretability**: Models remain largely uninterpretable - need explainability methods
2. **Threshold Selection**: Fixed threshold (0.5) suboptimal for class imbalance
3. **Training Budget**: Limited epochs may not fully exploit model capacity
4. **Pretraining**: ImageNet pretraining may not be optimal for microscopy images
5. **Resolution**: 256×256 may limit detection of fine-grained patterns

Future directions:
- Self-supervised pretraining on unlabeled microscopy images
- Transformer-based architectures for long-range dependencies
- Adaptive, class-specific thresholds
- Higher resolution inputs
- Attention maps and attribution methods

## References

### Dataset
- Human Protein Atlas Single-Cell Classification Competition
- Karlsson et al. (2021). A single-cell type transcriptomics map of human tissues. *Science Advances*

### Key Papers
- Ouyang et al. (2022). Analysis of the Human Protein Atlas Image Classification competition. *Nature Methods*
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*
- Brock et al. (2021). High-Performance Large-Scale Image Recognition Without Normalization. *ICML*
- Lin et al. (2017). Focal Loss for Dense Object Detection. *ICCV*
- Yun et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*

### Repository
- **GitHub**: https://github.com/aagamb/cs5787
- All code and experiments are publicly available for reproducibility

---

**Note**: This README is based on the comprehensive ablation study described in the full research paper. For detailed experimental results, error analysis, and methodological discussions, please refer to the complete report.

## Citation

If you use this code or reference this work, please cite:

```
Aagam Bakliwal, Shrey Verma, and Anushka Naik. 2025. 
A Comparative Study on Deep Learning Architectures for Protein Localization. 
ACM Conference Proceedings.
```

---


