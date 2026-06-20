# Fault Diagnosis for Rotating Machinery

This repository contains a time-series machine learning project for rotating machinery fault diagnosis. The project explores sensor-signal preprocessing, frequency-domain analysis, time-frequency feature extraction, and deep learning models for classifying machinery fault patterns.

## Project Overview

The goal of this project is to build an experimental pipeline for gearbox and wind-turbine fault diagnosis using vibration or sensor signals.

The workflow covers:

- Signal segmentation and standardization
- FFT-based frequency-domain analysis
- CWT-based time-frequency representation
- Baseline machine learning models
- 1D CNN / ResNet / Transformer-style model comparison
- Confusion-matrix evaluation and error analysis

## Repository Structure

| Folder / File | Description |
|---|---|
| `analysis/` | Exploratory analysis and signal-processing scripts |
| `experiments/` | Model experiments and comparison scripts |
| `confusion_matrix.png` | Classification result visualization |
| `gearset_analysis.png` | Gearset signal analysis result |
| `line_plot.png` | Time-series signal visualization |
| `PYTHON_FILES.md` | Summary of Python files in the repository |

## Methods

- Time-series preprocessing
- Fast Fourier Transform (FFT)
- Continuous Wavelet Transform (CWT)
- Traditional machine learning baselines
- 1D CNN / ResNet-style neural networks
- Transformer-based sequence modeling
- Confusion matrix and class-level error analysis

## Skills Demonstrated

- Time-series machine learning
- Signal processing
- Fault diagnosis
- Feature extraction
- Deep learning model comparison
- Python-based experimental workflow

## Notes

This project is an experimental research and portfolio project. The current version focuses on model exploration, preprocessing design, and diagnostic workflow construction rather than production deployment.
