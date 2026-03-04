# Chest X-Ray Tampering Detection Pipeline

This repository provides the **complete workflow used to build, tamper, and evaluate chest X-ray datasets** for tampering detection experiments using lightweight deep learning models.

The pipeline includes:

* Dataset preparation
* Cleaning corrupted images
* Selecting a balanced subset
* Generating tampered images
* Training detection models
* Evaluating on **unseen datasets**
* Producing **Grad-CAM visual explanations**

The **combined tampering pipeline** is the main method used in this project.

---

# 1. Datasets

## 1.1 NIH ChestX-ray8 (CXR8)

Primary dataset used for experiments.

Dataset link:
https://nihcc.app.box.com/v/ChestXray-NIHCC

Paper reference:
Wang et al., *ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization*, CVPR 2017.

After downloading the dataset you will obtain archive files that extract into folders like:

```
images_001/
images_002/
images_003/
...
images_012/
```

Each folder contains approximately **5,000 chest X-ray images**.

---

## 1.2 LIDC-IDRI Dataset

The **LIDC-IDRI dataset** is used as an additional unseen dataset for generalization testing.

Dataset link:
https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

Provided by **The Cancer Imaging Archive (TCIA)**, this dataset contains annotated lung CT scans.

Images from this dataset are used **only for unseen testing experiments**.

---

# 2. Preparing the CXR8 Dataset

## Step 1 — Extract the dataset archives

After downloading CXR8 you will get folders such as:

```
images_001/
images_002/
images_003/
...
```

Extract all archives and collect the image folders.

Example structure:

```
cxr8_raw/
   images_001/
   images_002/
   images_003/
   ...
```

---

## Step 2 — Merge images into one folder

Move all images into a single directory:

```
cxr8_all_images/
```

This folder should contain all CXR8 images.

---

# 3. Cleaning the Dataset

Run the cleaning script:

```
python clean.py
```

The script:

• Removes corrupted images
• Removes blank images
• Removes unreadable files

After execution a new dataset will be created:

```
cxr8_clean/
```

This directory contains only **valid chest X-ray images**.

---

# 4. Selecting Images for Experiments

Run:

```
python pick.py
```

This script:

• Randomly selects **20,000 images**
• Copies them to the experiment dataset

Output directory:

```
dataset/original/
```

This folder contains the **20k clean images used in the experiments**.

---

# 5. Generating Tampered Images

Tampered images are created using the **combined tampering generator**.

Run:

```
python combined.py
```

This script generates a balanced dataset containing:

| Class                     | Images |
| ------------------------- | ------ |
| Original                  | 5000   |
| Copy-move tampering       | 5000   |
| Patch insertion tampering | 5000   |
| Intensity modification    | 5000   |

Total images generated: **20,000**

The script also generates **ground truth masks**.

Output structure:

```
dataset/
   original/
   tampered/
   masks/
```

The **combined pipeline is the main tampering generator used in this project**, therefore separate tampering scripts are not required for the main experiments.

---

# 6. Training Models

After generating the dataset, training can be started.

Example training workflow:

```
python lbp_svm_glcm.py
python mobilenet_cnn_full.py
python fine_tune_mobilenetv2.py
```

These scripts will:

• Train baseline classical models
• Train MobileNetV2 tampering detector
• Fine-tune the last layers of the network

The trained model weights will be saved as:

```
model_name.keras
```

---

# 7. Testing on Unseen Data

To evaluate model generalization, testing is performed on **unseen datasets**.

The unseen dataset must be prepared using the **same preprocessing pipeline**:

1. Download new images
2. Run `clean.py` if necessary
3. Select images
4. Generate tampering using `combined.py`

Unseen datasets used in experiments include:

• Fresh **CXR8 images not used during training**
• Images from the **LIDC-IDRI dataset**

---

## Running Evaluation

After preparing unseen datasets run:

```
python evaluate_and_plot_final.py
```

This script:

• Loads the saved **trained model (.keras)**
• Runs inference on unseen data
• Computes performance metrics
• Generates evaluation plots

---

# 8. Generating Grad-CAM Visualizations

Grad-CAM is used to visualize **which regions of the image influenced the tampering prediction**.

To generate Grad-CAM images:

1. Select a **tampered image**
2. Ensure the trained model `.keras` file is available
3. Run:

```
python gradcam.py
```

The script will:

• Load the trained model weights
• Generate Grad-CAM heatmaps
• Overlay attention maps on the input image

Output images are saved in:

```
gradcam_results/
```

---

# 9. Full Experiment Pipeline

```
Download CXR8 Dataset
        ↓
Extract images_001 … images_012
        ↓
Merge all images
        ↓
clean.py
        ↓
cxr8_clean
        ↓
pick.py
        ↓
dataset/original (20k images)
        ↓
combined.py
        ↓
tampered images + masks
        ↓
Model training
        ↓
Evaluation on unseen datasets
        ↓
Grad-CAM visualization
```

---

# 10. Notes

• The **combined tampering pipeline** is the primary dataset generator used in this research.
• Separate tampering scripts exist but are not required for the main experiment pipeline.
• All tampering operations are **synthetic and used only for research purposes**.
• Ground-truth masks are automatically generated for evaluation and visualization.

---
