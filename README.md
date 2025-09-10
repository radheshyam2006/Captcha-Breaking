# CAPTCHA Breaking Challenge

A comprehensive machine learning project that implements three different approaches to break CAPTCHA systems, ranging from basic image classification to advanced color-conditional text recognition using computer vision.

## Project Overview

This project demonstrates the evolution of CAPTCHA-breaking techniques through three distinct challenges:

1. **Basic CAPTCHA Solver**: Simple CNN-based image classification
2. **Multi-Difficulty CAPTCHA Solver**: Advanced OCR using CNN + LSTM with CTC loss
3. **Color-Conditional CAPTCHA Solver**: Computer vision-enhanced OCR with conditional text processing

## Project Structure

```
can_you_break_the_captcha-main/
├── codes/
│   ├── basic_captcha_generator.py
│   ├── basic_captcha_solver.py
│   ├── multi_difficulty_captcha_generator.py
│   ├── multi_difficulty_captcha_solver.py
│   ├── color_conditional_captcha_generator.py
│   └── color_conditional_captcha_solver.py
├── datasets/
│   ├── basic_dataset/
│   ├── multi_difficulty_dataset/
│   └── color_conditional_dataset/
└── README.md
```

## Setup and Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies Installation

```bash
# Install required packages
pip install torch torchvision pandas pillow opencv-python numpy
```

### Alternative installation:
```bash
pip install -r requirements.txt
```

Required libraries:
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `pandas` - Data manipulation
- `pillow` - Image processing
- `opencv-python` - Computer vision (for color-conditional solver)
- `numpy` - Numerical computations

## Quick Start Guide

### 1. Basic CAPTCHA Challenge

**Generate Dataset:**
```bash
cd codes/
python basic_captcha_generator.py
```
- Creates simple handwritten-style images
- Saves images in `basic_dataset/` directory
- Generates `labels.csv` with image-text mappings

**Train and Test Model:**
```bash
python basic_captcha_solver.py
```
- Uses CNN for image classification
- Prompts for dataset directory
- Outputs predictions to `predictions.csv`

### 2. Multi-Difficulty CAPTCHA Challenge

**Generate Dataset:**
```bash
python multi_difficulty_captcha_generator.py
```
- Creates both **easy** and **hard** CAPTCHA images
- Easy: Clean text with minimal noise
- Hard: Multiple fonts, colors, and noise effects

**Train OCR Model:**
```bash
python multi_difficulty_captcha_solver.py
```
- Implements CNN + LSTM architecture
- Uses CTC (Connectionist Temporal Classification) loss
- Prompts for:
  - Dataset directory path
  - Number of training epochs

### 3. Color-Conditional CAPTCHA Challenge

**Generate Dataset:**
```bash
python color_conditional_captcha_generator.py
```
- Creates images with **green** or **red** backgrounds
- Text direction depends on background color
- Green background = normal text
- Red background = reversed text logic

**Train Enhanced Model:**
```bash
python color_conditional_captcha_solver.py
```
- Combines OCR with computer vision
- Detects background color using OpenCV
- Conditionally processes text based on color detection

## Model Architectures

### Basic CAPTCHA Solver
- **Architecture**: Simple CNN with fully connected layers
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam
- **Task**: Multi-class image classification

### Multi-Difficulty CAPTCHA Solver
- **Architecture**: CNN (feature extraction) + LSTM (sequence modeling)
- **Loss Function**: CTC Loss for sequence prediction
- **Input Size**: 32×128 grayscale images
- **Output**: Variable-length text sequences

### Color-Conditional CAPTCHA Solver
- **Architecture**: CNN + LSTM + Computer Vision
- **Additional Feature**: Background color detection
- **Color Analysis**: RGB channel separation and intensity comparison
- **Conditional Logic**: Text reversal based on red/green background

## Technical Approaches

### 1. Basic Approach
```python
# Simple CNN classification
- Image → CNN → Feature Vector → Classifier → Predicted Class
```

### 2. Advanced OCR Approach
```python
# Sequence-to-sequence learning
- Image → CNN → Feature Maps → LSTM → Character Probabilities → CTC Decode
```

### 3. Computer Vision Enhanced Approach
```python
# Multi-modal processing
- Image → [CNN+LSTM for OCR] + [Color Detection] → Conditional Text Processing
```

## Performance Features

- **Data Augmentation**: Random fonts, colors, and noise
- **Regularization**: Dropout layers to prevent overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Validation Monitoring**: Real-time accuracy tracking
- **CTC Decoding**: Handles variable-length sequences without alignment

## Customization Options

### Dataset Generation Parameters:
- **Image dimensions**: Adjustable width/height
- **Font varieties**: Multiple font families
- **Noise levels**: Configurable difficulty
- **Color schemes**: Background and text colors
- **Text length**: Variable character sequences

### Training Parameters:
- **Batch size**: Configurable for different hardware
- **Learning rate**: Adjustable optimization speed
- **Epochs**: Training duration control
- **Validation split**: Train/validation/test ratios

## Expected Outputs

1. **Dataset Generation**: 
   - Image files (.png)
   - Labels CSV file
   - Directory structure creation

2. **Model Training**:
   - Epoch-wise loss and accuracy
   - Model convergence metrics
   - Final test accuracy

3. **Predictions**:
   - `predictions.csv` with image names and predicted text
   - Character-level accuracy scores