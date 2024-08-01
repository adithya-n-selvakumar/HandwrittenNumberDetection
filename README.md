## Overview

This project implements a Convolutional Neural Network (CNN) that achieves over 99.5% accuracy on the MNIST dataset using PyTorch. The model incorporates several advanced techniques to enhance performance and generalization.

## Features

- Custom CNN architecture with multiple convolutional layers
- Data augmentation techniques for improved generalization
- Batch normalization and dropout for regularization
- Learning rate scheduling using CosineAnnealingLR
- Comprehensive training and evaluation pipeline
- Visualization of training metrics and confusion matrix

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- seaborn
- scikit-learn

## Installation

1. Clone this repository:
git clone https://github.com/adithya-n-selvakumar/HandwrittenNumberDetection.git
cd HandwrittenNumberDetection

2. Install the required packages:
!pip install torch torchvision
!pip install seaborn

## Usage

Run the main script to train the model:
This will:
1. Load and preprocess the MNIST dataset
2. Train the CNN model
3. Evaluate the model on the test set
4. Save the best model weights
5. Generate and save training metrics and confusion matrix plots

## Model Architecture

The CNN architecture consists of:
- 3 convolutional layers with batch normalization and ReLU activation
- Max pooling after each convolutional layer
- 2 fully connected layers with dropout

## Results

The model typically achieves over 99.5% accuracy on the MNIST test set within 100 epochs. Actual results may vary slightly due to the random nature of initialization and data augmentation.

## Visualization

The script generates two plots:
1. `training_metrics.png`: Shows the training/test loss, test accuracy, and learning rate over epochs
![image](https://github.com/user-attachments/assets/978a1099-5b8d-4093-8337-a2326df29475)

2. `confusion_matrix.png`: Displays the confusion matrix for the best model on the test set
![image](https://github.com/user-attachments/assets/1fc891ba-bc2f-4c33-9930-b0474e4ebcb2)

## Author

Adithya N. Selvakumar

## Acknowledgments

- The MNIST dataset creators and maintainers
- The PyTorch team for their excellent deep learning framework
