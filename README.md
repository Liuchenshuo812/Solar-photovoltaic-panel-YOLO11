# PLE-YOLO: A Multi-Scale Attention-Enhanced Framework for Solar Photovoltaic Panel Defect Detection

## Project Overview

Title：《PLE-YOLO: A Multi-Scale Attention-Enhanced Framework for Solar Photovoltaic Panel Defect Detection》

This project proposes an improved YOLOv11s model for solar photovoltaic (PV) panel defect detection. By integrating a **P2 high-resolution detection head**, a **Local Feature Enhancement Module (LFEM)**, and an **Efficient Local Attention (ELA)** module, the model achieves high-precision detection of three PV panel conditions: **Dirty**, **Clean**, and **Broken**. The proposed framework aims to provide a lightweight and efficient solution for intelligent PV plant inspection.

## 1. Research Background and Model Positioning

Photovoltaic panels inevitably suffer from defects such as dust accumulation, hot spots, and micro-cracks during long-term outdoor operation, which reduce power generation efficiency and pose safety risks. Traditional inspection methods (e.g., EL and infrared thermography) rely heavily on manual interpretation and lack generalization. This project aims to construct an efficient deep learning model based on an improved YOLOv11s architecture, providing a fast, accurate, and deployable solution for automated PV panel defect detection in complex environments.

## 2. Core Innovations

1. **P2 Detection Branch:**
   Adds an independent P2 detection branch at the 4× downsampling layer of the backbone, constructing a P2-P5 four-scale prediction architecture. This effectively preserves boundary and texture information for extremely small defects (e.g., fine cracks and tiny dirt particles) without significantly increasing the number of parameters.

2. **LFEM (Local Feature Enhancement Module):**
   Integrated into the deep layers of the neck network (after the P5 layer). It employs a four-branch parallel structure:
   - Branch 1: 1×1 Conv
   - Branch 2: 3×3 Conv
   - Branch 3: 3×3 dilated Conv (dilation=3)
   - Branch 4: 3×3 depthwise Conv
   The outputs are concatenated, channel-shuffled, fused by a 1×1 Conv, and added to the input via a residual connection. This design captures multi-scale local context and enhances spatial structural expression.

3. **ELA (Efficient Local Attention):**
   Embedded at the front end of the detection head. It performs 1D average pooling along horizontal and vertical directions, followed by depthwise convolution and a gating mechanism (Sigmoid). The input is multiplied element-wise with the generated attention weights. This lightweight design preserves fine-grained spatial information, strengthens key defect areas, and suppresses complex background noise.

## 3. Efficiency and Accuracy

The model maintains a lightweight architecture while achieving superior performance. On the main SPHERE dataset, the proposed PLE-YOLO achieves **97.99% mAP@0.5**, outperforming the baseline YOLOv11s and other mainstream detectors. By integrating P2, LFEM, and ELA, the model forms a progressive multi-scale feature refinement chain, delivering high accuracy in both detection and localization, making it highly suitable for deployment in real-world industrial PV inspection.

### 3.1 Experimental Dataset

This study uses two independent datasets:
- **Main Dataset (Open-source):** The "SPHERE Solar Panel Hidden Defect Evaluation for Renewable Energy" dataset, containing 6,079 annotated images across three categories: Dirty (2,245), Clean (2,130), and Broken (1,704). Split into train/val/test at a ratio of approximately 7:2:1. Available at: [https://www.kaggle.com/dsv/11567509](https://www.kaggle.com/dsv/11567509)
- **Generalization Test Dataset (Open-source):** The "Solar Panel Images Clean and Fault Images" dataset, containing 452 images across three categories: Dusty (190), Clean (193), and Physical Damage (69). Used for cross-dataset generalization validation. Available at: [https://www.kaggle.com/datasets/pythonafroz/solar-panel-images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images)

### 3.2 Dataset Structure

The dataset folder is organized as follows:

```plaintext
dataset_SPHERE/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```
A data.yaml file is essential for training. It defines the absolute paths to the train, valid, and test directories, along with the mapping of class indices to their names (e.g., 0: Dirty, 1: Clean, 2: Broken).

## 4. Experimental Environment Configuration

### 4.1 Dependency Installation

We recommend using Anaconda to create a virtual environment.

```
1. Create and activate a virtual environment
conda create -n pv-yolo11 python=3.10
conda activate pv-yolo11
2. Install PyTorch (compatible with CUDA 12.8)
pip install torch==2.7.0 torchvision==0.22.0
3. Install other dependencies
pip install -r requirements.txt
```
### 4.2 Hardware Requirements

OS: Windows 10 x64

Python: 3.10

CUDA: 12.8, cuDNN: 8.7

Framework: PyTorch 2.7.0 / Ultralytics

Hardware Tested On: AMD Ryzen 9 8945HX, 32GB RAM, NVIDIA GeForce RTX 5060


## 5. Experimental Results

Performance comparison between the proposed model and baseline YOLOv11s:

|Category|mAP@0.5|
|-|-|
|YOLOv12|97.05%|
|Proposed|97.99%|

Note: The P2 branch significantly improves small defect detection. LFEM captures rich multi-scale local textures to combat resolution degradation, while ELA effectively suppresses redundant background noise to highlight actual defect areas.


## 6. Code Usage Instructions

### 6.1 Model Training
Run the main.py script to start training. The model configuration is defined in the YAML file.
```
python main.py
```
Main configuration items (can be modified in main.py):

|Parameter|Value|
|-|-|
|imgsz|320|
|batch|16|
|epochs|150|
|lr0|0.0002|
|optimizer|AdamW|

## 7. Project File Structure

```
Solar-Photovoltaic-Panel-YOLO11/
├── dataset_SPHERE/             # Dataset directory (contains train, valid, test)
├── main.py                     # Main training and evaluation script
├── yolov11_P2_LFEM_ELA.yaml    # Improved YOLOv11 network configuration
├── train_batch0.jpg            # Visualization of training batch 0
├── train_batch1.jpg            # Visualization of training batch 1
├── train_batch2.jpg            # Visualization of training batch 2
└── README.md
```
Note: The implementations of the LFEM and ELA modules are integrated within the model configuration and main.py.


## 8. Training Visualization

The following images show the visualizations of the training process and cross-domain generalization performance on external visible light images (Confidence > 90%):
https://train_batch0.jpg/
https://train_batch1.jpg/
https://train_batch2.jpg/

## 9. Known Issues and Precautions

1. Computational Complexity: The additional P2 high-resolution detection branch inevitably increases computational complexity and memory usage. Edge devices may need structural pruning or low-bit quantization for optimal performance.

2. CUDA Compatibility: If you encounter CUDA incompatibility, ensure your PyTorch version matches your local CUDA version (12.8 recommended).

3. Memory Usage: If you encounter out-of-memory errors, reduce batch size from 16 to 8 or 4.

4. Domain Shift: While the model shows strong cross-domain adaptability on visible light images, extreme lighting, large-scale occlusion, or cross-modal scenes may still require further fine-tuning or the introduction of linear attention mechanisms.

## 10. References and Contact Information

### 10.1 Reference Method

The paper is currently in the submission stage. Currently available for temporary reference:
@article{liu2026pleyolo,
  title={PLE-YOLO: A Multi-Scale Attention-Enhanced Framework for Solar Photovoltaic Panel Defect Detection},
  author={Liu, Chenshuo and Xu, Laixiang and Xu, Jinying and Zhao, hao},
  journal={（待录用后补充）},
  year={2026},
  note={Manuscript submitted for publication}
}

### 10.2 Contact Information

If you encounter code running issues or academic exchange needs, please contact:
Email: liuchenshuohuuc@yeah.net

GitHub Issue: Submit an issue directly to this repository and we will respond within 1-3 business days.

Repository Link: https://github.com/Liucheshuo812/Solar-photovoltaic-panel-YOLO11.git



