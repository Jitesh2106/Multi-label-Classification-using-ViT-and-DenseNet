# Multi-label Classification using ViT and DenseNet with Transfer Learning

This repository contains code for a multi-label classification task utilizing Vision Transformer (ViT) and DenseNet architectures with transfer learning. The task involves classifying 104 types of flowers based on their images drawn from five different public datasets. The dataset contains imperfections such as images of flowers in odd places or as a backdrop to modern machinery, presenting a challenge for the classifier to identify the flowers accurately.

## Dataset Description

The dataset is provided in TFRecord format, which is commonly used in TensorFlow for optimal training performance. Each file contains the following information for multiple images:

- **id**: Unique ID for each sample.
- **label**: Class of the sample (for training data).
- **img**: Actual pixels in array form.

https://www.kaggle.com/competitions/tpu-getting-started/data

## Architecture

The project utilizes two architectures for multi-label classification:

1. **Vision Transformer (ViT)**: A transformer-based model specifically designed for image classification tasks. It breaks down an image into fixed-size patches, flattens them into sequences, and applies self-attention mechanism for feature extraction.

2. **DenseNet**: A densely connected convolutional neural network architecture. DenseNet connects each layer to every other layer in a feed-forward fashion, allowing for feature reuse and efficient learning.

## Transfer Learning

Both ViT and DenseNet architectures can benefit from transfer learning. Pre-trained models, can be fine-tuned on the flower dataset to improve performance, reduce training time, and handle limited data scenarios effectively.

## Usage

1. **Data Preparation**: The TFRecord files can be loaded and preprocessed using TensorFlow's data loading utilities. Data augmentation techniques are applied for better generalization.

2. **Model Training**: Fine-tune the pre-trained ViT and DenseNet models using the prepared dataset. It is recommended to use GPUs or TPUs for faster training.

3. **Model Evaluation**: Evaluate the trained models on the validation set to assess their performance. 

## Requirements

- TensorFlow
- NumPy
- Pandas
- Matplotlib (for visualization, optional)
- GPU or TPU (recommended for faster training)

## Acknowledgements

This project is based on TensorFlow and leverages the power of Vision Transformer and DenseNet architectures for multi-label classification with transfer learning. The dataset is sourced from a public competition, and additional documentation and resources are provided by the competition organizers.
