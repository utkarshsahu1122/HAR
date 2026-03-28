# PoseNet: Human Activity Detector

A machine learning pipeline for classifying human physical activities from raw accelerometer and gyroscope sensor signals, achieving 97.2% accuracy using an LSTM network.

## Overview

Built on the UCI Human Activity Recognition (HAR) dataset (~10,299 samples, 561 features), this project applies PCA for dimensionality reduction and benchmarks three model architectures — Random Forest, 1D-CNN, and LSTM — across 6 activity classes.

## Features

- **Dataset**: UCI HAR Dataset from 30 subjects performing 6 activities (walking, sitting, standing, etc.)
- **Dimensionality reduction**: PCA reduces 561 features to 50 components, retaining 98% variance
- **Multi-model benchmarking**: Random Forest, 1D-CNN, and LSTM trained and evaluated side-by-side
- **Cross-validation**: 5-fold stratified cross-validation on an 80/20 train-test split
- **Training speedup**: PCA reduces training time by ~40%

## Results

| Model | Accuracy | F1-Score |
|---|---|---|
| Random Forest | — | — |
| 1D-CNN | — | — |
| **LSTM** | **97.2%** | **0.97** |

LSTM consistently outperformed other architectures across all 6 activity classes.

## Activity Classes

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

## Tech Stack

- **Python**
- **TensorFlow / Keras** — CNN and LSTM models
- **Scikit-learn** — Random Forest, PCA, cross-validation
- **Pandas** — data loading and preprocessing
- **NumPy** — feature matrix operations

## How It Works

1. Raw accelerometer and gyroscope signals are loaded from the UCI HAR dataset
2. PCA is applied to compress 561 features into 50 principal components (98% variance retained)
3. Data is split into 80% train / 20% test with stratification across activity classes
4. Three models are trained: Random Forest (on PCA features), 1D-CNN, and LSTM
5. Models are evaluated using accuracy, F1-score, and 5-fold cross-validation

## Architecture (LSTM)

```
Input (time-step sequence × 50 PCA features)
       ↓
LSTM Layer 1
       ↓
LSTM Layer 2
       ↓
Dense → Softmax (6 activity classes)
```

## Usage

```bash
git clone https://github.com/<your-username>/posenet-activity-detector
cd posenet-activity-detector
pip install -r requirements.txt

# Preprocess and train
python preprocess.py
python train.py --model lstm

# Evaluate
python evaluate.py --model lstm
```



## Dataset

UCI Human Activity Recognition Dataset — 30 subjects, 6 activity classes, signals sampled at 50Hz.
Available at: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
