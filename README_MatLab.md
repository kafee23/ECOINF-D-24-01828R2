# Mango Leaf Disease Detection – Feature Extraction (MATLAB)

This repository contains MATLAB scripts and documentation for the handcrafted spatial feature extraction pipeline used in our mango leaf disease detection model. The code forms the basis of the first stage in our two-stage model and supports efficient feature generation for machine learning classifiers.

## Overview

The process extracts robust, multi-scale texture features from mango leaf images using Local Directional Pattern (LDP) and Local Directional Pattern Variance (LDPv). It comprises five major steps, from image preprocessing to the final concatenated feature matrix, used to train traditional classifiers and knowledge distillation models.

## Prerequisites

- MATLAB (with Image Processing Toolbox)
- Raw RGB mango leaf images (see [MangoLeafBD dataset](https://data.mendeley.com/datasets/hxsnvwty3r/1))
- Directory structure consistent with the scripts
- `.mat` script files provided in this repository

## Feature Extraction Pipeline

### Step 1: Image Preprocessing

**Scripts:** `img_resize.mat`, `rgb_gray.mat`  
- Resizes all input images to a standard dimension  
- Converts RGB images to grayscale

### Step 2: Directional Value Extraction

**Script:** `masking_pic.mat`  
- Applies Kirsch edge masks to derive directional information from grayscale images  
- Outputs 8-directional response matrices

### Step 3: LDP and LDPv Feature Generation

**Scripts:** `LDP.mat`, `LDPv.mat`  
- Computes LDP codes (top-3 edge directions) for texture encoding  
- Calculates LDPv to measure local variance in edge strength  
- Both are computed for block sizes: 3×3, 6×6, and 9×9

### Step 4: Multi-Scale Mean Feature Computation

**Script:** `datasetCode.mat`  
- Computes mean features within blocks (3×3, 6×6, 9×9) for each LDP and LDPv matrix  
- Output: Six feature vectors per image (3 LDP + 3 LDPv)

### Step 5: Feature Concatenation

**Script:** `concate_allfeature.mat`  
- Concatenates all six mean feature vectors:  
  - `LDP_3x3`, `LDP_6x6`, `LDP_9x9`  
  - `LDPv_3x3`, `LDPv_6x6`, `LDPv_9x9`  
- Final feature vector represents each image with combined spatial descriptors

> ⚠️ Note: The final feature vector only includes mean values from Step 4, computed from LDP and LDPv for all three block sizes. Raw directional values (from Step 2) are not included in the concatenation at this stage.

## Output

- A `.mat` file containing the final feature matrix, with one row per image  
- Ready for use with machine learning classifiers like SVM, KNN, or student models in the KD framework

## Citation

If you use this repository or the associated methodology in your research, please cite:

> Islam, M. M., et al. “A Two-Stage Model for Enhanced Mango Leaf Disease Detection Using An Innovative Handcrafted Spatial Feature Extraction Method and Knowledge Distillation Process.” _Ecological Informatics_, 2025.

## License

MIT License
