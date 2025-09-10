# TASK-0

# images for task-1
# Handwritten Dataset Generator

This project generates a synthetic handwritten dataset of images containing random words with various styles and noise. It uses the Python `PIL` library to create these images, applies random noise, capitalizes letters randomly, and stores the images and their corresponding labels in a CSV file.



## Approach

1. **Word Selection**: A list of predefined words is selected randomly.
2. **Image Generation**: Images are created with random text, font, and noise variations.
3. **Text Noise**: Noise is applied to the image to make the text appear more handwritten.
4. **Capitalization**: Random letters in the word are capitalized for variety.
5. **Font & Color Variation**: Different fonts and text colors are used to diversify the dataset.
6. **Image and Label Saving**: The images are saved in the `hdataset` folder, and a CSV file `labels.csv` records each image's name and text label.

## Dependencies

This project requires the following libraries:
- `Pillow`: For image generation and manipulation.
- `random`: For generating random numbers to apply variability.

Install the required libraries using:

```bash
pip install pillow
```

# commands to run
python task1imagegeneration.py

# images for task-2

# Handwritten Dataset Generator

This project generates a synthetic handwritten dataset consisting of both easy and hard images containing randomly generated words. It uses Python’s `PIL` (Pillow) library to create images, apply noise, capitalize random letters, and store them along with their labels in a CSV file.




## Approach

1. **Word Generation**: A random word is generated with a random length from the alphabet.
2. **Image Generation**: The script generates images (easy and hard) containing randomly selected words.
3. **Text Noise**: For the hard dataset, random noise is applied to make the text appear more handwritten.
4. **Font & Capitalization**: Different fonts are chosen randomly, and letters in words are randomly capitalized.
5. **Image and Label Saving**: The images are saved in the specified directory, and their corresponding labels are written into a CSV file.

## Dependencies

This project requires the following Python libraries:
- `Pillow` (PIL Fork) for image creation and manipulation.
- `random` for generating random values.

You can install the dependencies using:
# commands to run
python task2imagegeneration.py

```bash
pip install pillow
```



# Tak-3 images generation

This project generates synthetic handwritten images containing random words with customizable variations. The images are saved in a user-specified directory, and a CSV file is created with the corresponding image names and text labels.



## Approach

1. **Word Generation**: A random word is created with random length from the alphabet.
2. **Image Creation**: A blank image with random background colors (`green` or `red`) is created, and text is added using randomly selected fonts.
3. **Capitalization**: Random letters are capitalized.
4. **Noise Addition**: Optionally, background color and text are reversed when the background is red.
5. **CSV File**: Image names and corresponding text are written to `labels.csv`.

## Dependencies

This project uses the following Python libraries:

- `Pillow` for image manipulation.
- `random` for generating random values.

Install the dependencies using:

```bash
pip install pillow
```

python task1imagegeneration.py


# TASK-1

# Image Classification with CNN

This project uses Convolutional Neural Networks (CNN) to classify images based on their corresponding text labels. The dataset consists of images stored in a specified directory, and a CSV file containing the image names and their respective labels.

## Directory Structure





## Dependencies

This project requires the following libraries:

- **torch**: The core PyTorch library for building and training the neural network.
- **torchvision**: Provides datasets, model architectures, and image transformations.
- **pandas**: For handling CSV files and datasets.
- **PIL**: For loading and manipulating images.


# You can install the necessary dependencies by running:

pip install torch torchvision pandas pillow

-**My approach follows a supervised image classification pipeline using a CNN-based deep learning model in PyTorch.**

# Data Preprocessing:

-**The script starts by loading the labels.csv file which contains the image names and their labels.**
A mapping of labels to numerical indices is created for model training.
# Dataset Class:

The WordImageDataset class inherits from torch.utils.data.Dataset and is used to load images and their corresponding labels.
Each image is loaded using PIL and transformed using a predefined set of transformations (resize, tensor conversion, normalization).
# Model Architecture:

A simple CNN model with three convolutional layers followed by fully connected layers.
The model uses ReLU activation, max pooling, dropout for regularization, and softmax output for classification.
# Training Loop:

The model is trained using Cross-Entropy Loss and the Adam optimizer.
The accuracy and loss are printed after each epoch.
# Validation and Prediction:

After training, the model's performance is evaluated on the validation set.
The predicted labels are saved along with their corresponding image names in a CSV file.

python task1.py


# TASK-2

# OCR Model for Text Recognition

## Overview
This project implements an Optical Character Recognition (OCR) model using PyTorch. The model consists of a CNN for feature extraction followed by an LSTM for sequence modeling. It is trained using the Connectionist Temporal Classification (CTC) loss function.

## Directory Structure





## Dependencies
To install dependencies, run:
pip install -r requirements.txt


### Required Libraries
- Python 3.x
- torch
- torchvision
- pandas
- numpy
- PIL (Pillow)

## Running the Project
1. **Run the training script**
python task2.py

The script will prompt for:
- `Enter directory name:` (Path to the dataset)
- `Enter number of epochs:` (Number of training epochs)

2. **Model Evaluation**
- The model prints training loss and validation accuracy after each epoch.
- After training, it evaluates test accuracy.

## Approach
### 1. Dataset Preparation  
- Images are converted to grayscale and resized to `(32,128)`.
- Labels are mapped to integer sequences for training.
- Data is split into train (80%), validation (10%), and test (10%).

### 2. Model Architecture  
- A **CNN** extracts spatial features from the images.
- An **LSTM** processes these features sequentially.
- A **fully connected layer** maps outputs to character probabilities.

### 3. Training Strategy  
- **CTC loss** is used for training.
- The **Adam optimizer** with a learning rate scheduler is applied.
- Accuracy is computed using character-wise predictions.

### 4. Evaluation  
- The trained model is tested on unseen images.
- Predictions are compared to ground-truth labels.

## Prediction
To predict text from an image:
```python
predicted_text = predict_text("image_path.png", model)
print(predicted_text)
```
python task2.py


# TASK-3
 I t is totally same as TASK-2 just we have used the  I have just used computer vision remaining every thing is same as TASK-2
python task1.py