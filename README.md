# CoronarySegmentation

This repository contains code developed for training and evaluating deep learning models for **coronary artery segmentation** in CT angiography (CTCA), using the publicly available [ASOCA dataset](https://asoca.grand-challenge.org/). The approach integrates **centerline priors** into the learning process, with the goal of assessing whether this anatomical guidance can improve segmentation performance—particularly in challenging cases involving complex vessel morphology or pathology.

The pipeline supports full-volume preprocessing, 2D and 3D segmentation workflows, and is designed for flexibility in model architecture.

## 🧠 Centerline Prior

To enhance segmentation performance and encourage anatomical awareness, this project integrates **coronary centerline information** into the model training pipeline in two key ways: intelligent cropping and attention-based input encoding.

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



## ⚙️ Requirements & Setup

To get started, clone the repository and set up your environment using one of the following options:

### 🔁 1. Clone the Repository

```bash
git clone https://github.com/FrancescoLeni/CoronarySegmentation.git
cd CoronarySegmentation

```bash
pip -r install requirements.txt






