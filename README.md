# CoronarySegmentation

This repository contains code developed for training and evaluating deep learning models for **coronary artery segmentation** in CT angiography (CTCA), using the publicly available [ASOCA dataset](https://asoca.grand-challenge.org/). The approach integrates **centerline priors** into the learning process, with the goal of assessing whether this anatomical guidance can improve segmentation performance—particularly in challenging cases involving complex vessel morphology or pathology.

The pipeline supports full-volume preprocessing, 2D and 3D segmentation workflows, and is designed for flexibility in model architecture.

## 🧠 Centerline Prior

To enhance segmentation performance and encourage anatomical awareness, this project integrates **coronary centerline information** into the model training pipeline in two key ways: intelligent cropping and attention-based input encoding.

<p align="center">
  <img src="repo_data/cropping.gif" alt="cropping" width="320"/>
  <img src="repo_data/fix%20grid.gif" alt="grid" width="320"/>
</p>


---

### 🗂️ ROI Cropping

Instead of relying on traditional fixed-grid slicing strategies, we leverage the **coronary centerline** to identify the most informative regions within each CT slice.

- We cluster the centerline points along the axial volume using spatial clustering (e.g., k-means), obtaining centroids that represent high-content regions along the vessels.
- Each 2D slice is then cropped around these **centerline-derived centroids**, rather than using arbitrary or equidistant grid crops.
- This strategy **maximizes anatomical relevance per crop**, enabling the model to focus on coronary-rich regions rather than large backgrounds or irrelevant structures.
- As a result, we achieved **comparable or superior performance using less than half the training data**, thanks to more efficient data sampling guided by anatomical priors.

---

### 🎯 Attention

To further emphasize coronary structures during training, we introduce a form of **spatial attention** by modifying the input representation:

- Each original grayscale CT slice is augmented with a second channel.
- In this additional channel, the pixels belonging to the **centerline mask** are artificially enhanced (brightened) to guide the model’s focus.
- This dual-channel input allows the network to better **attend to the vascular structures**, particularly in ambiguous or low-contrast regions, without modifying the original image intensities.

Together, these centerline-driven strategies inject prior anatomical knowledge into the training process, resulting in more accurate, focused, and data-efficient coronary segmentation.

---

## 🧠 Model Architecture

This work explores both **2D** and **3D U-Net architectures** for coronary artery segmentation. The goal was to determine whether volumetric context (via 3D convolutions) and anatomically-aware cropping could enhance segmentation performance on the ASOCA dataset.

- **2D U-Net**: Applied slice-by-slice with or without centerline-guided cropping.
- **3D U-Net**: Processes volumetric patches, enabling the model to learn spatial continuity across adjacent slices.

Both models were trained under two different input sampling strategies:
- **Grid**: Traditional uniform slicing across the volume.
- **Crop**: Centerline-guided region-of-interest cropping (see [Centerline Prior](#-centerline-prior)).

### 🔬 Key Findings

The **3D U-Net architecture** combined with **centerline-guided cropping** achieved the best overall performance across all evaluated metrics. This supports the hypothesis that anatomical priors and 3D spatial context are beneficial for fine-grained vessel segmentation.

---

## 📊 Quantitative Results

| Metric     | 3D U-Net + Crop | 3D U-Net + Grid |
|------------|------------------|------------------|
| **Dice**        | **0.87**          | 0.80             |
| **Accuracy**    | **0.94**          | 0.90             |
| **Precision**   | **0.84**          | 0.83             |
| **Recall**      | **0.94**          | 0.92             |

---

## 🎥 Qualitative Results

<p align="center">
  <img src="repo_data/3d%20result pred.gif" alt="3D U-Net + Crop" width="400" />
  <img src="repo_data/3d%20result.gif" alt="3D U-Net + Grid" width="400" />
</p>

> 📌 All values are averaged across the test set using 3D predictions with post-processing.

---


## ⚙️ Requirements & Setup

To get started, clone the repository and set up your environment:

### 🔁 1. Clone the Repository

```
git clone https://github.com/FrancescoLeni/CoronarySegmentation.git
cd CoronarySegmentation
```

```
pip -r install requirements.txt
```

## 🧪 Experiments

### 📂 Dataset

The dataset used in this study is the **[ASOCA (Automatic Segmentation of Coronary Arteries) dataset](https://asoca.grand-challenge.org/)**. You must request access through the challenge page. Once downloaded, please organize the dataset as follows inside a `dataset/` directory:


```
dataset/
├── train/
│ ├── Normal/
│ │ └── CTCA/<PatientID>/
│ └── Diseased/
│ └── CTCA/<PatientID>/
├── val/
│ ├── Normal/
│ │ └── CTCA/<PatientID>/
│ └── Diseased/
│ └── CTCA/<PatientID>/
└── test/
├── Normal/
│ └── CTCA/<PatientID>/
└── Diseased/
└── CTCA/<PatientID>/
```

### 🧠 Pretrained Weights

You can download pretrained model weights from the following link:

🔗 **[Download Pretrained Weights](https://www.dropbox.com/scl/fo/)**  

---

### 🚀 Running Training

Use the following command to train a model (e.g., 3D U-Net with centerline cropping):

```
python train.py --model unet3d_crop --epochs <n° epochs> --batch_size <batch size>
```

Use the following command to test a trained model:

```
python test.py --model unet3d_crop --weights <path_to_weights>
```

## 🙏 Acknowledgments

We would like to thank the authors of the following repositories for their contributions, which were instrumental to this work:

- [DatasetUtilities](https://github.com/AAMIASoftwares-research/DatasetUtilities): for essential tools to load, align, and manage coronary CT volumes and annotations.
- [HCATNetwork](https://github.com/AAMIASoftwares-research/HCATNetwork): for implementing helpful utilities to process and manage centerline graph representations.

