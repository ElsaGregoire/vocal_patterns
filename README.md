# Vocal Patterns App: Machine Learning Functionality

## Overview

Our app, Vocal Patterns, utilizes advanced machine learning (ML) techniques to analyze vocal recordings. It can determine whether a user has sung a scale, arpeggio, or other vocal exercises. This capability is powered by a Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) layers, implemented using Keras and TensorFlow.

## ML Model Architecture

The core of our ML functionality is a Sequential model built using Keras. The model architecture comprises:

1. **Convolutional Layers**: Four convolutional layers with varying filter sizes and strides, each followed by batch normalization. These layers are crucial for extracting spatial features from the input spectrograms of vocal recordings.

2. **Bidirectional LSTM Layers**: These layers process the output of the convolutional layers in both forward and backward directions, capturing the temporal dynamics in the vocal recordings.

3. **Output Layer**: A dense layer with softmax activation to classify the input into three categories - scale, arpeggio, or other.

## Data Preprocessing

The `preprocessor.py` script handles data preparation, including:
- Generating Mel spectrograms from audio files.
- Augmenting data with various techniques like noise addition, stretching, and slicing waveforms.
- Normalizing spectrograms for consistent model input.

## Training and Evaluation

The `train` function in `main.py` orchestrates the model training process. Key steps include:
- Splitting data into training and validation sets.
- Compiling the model with Adam optimizer and categorical crossentropy loss.
- Fitting the model to the training data with early stopping based on validation loss.
- Evaluating model performance on a separate validation dataset.

## Model Saving and Predictions

After training, the model is saved to MLFlow for future use. Predictions are made by processing new vocal recordings through the same preprocessing pipeline and feeding the resulting spectrograms to the trained model.


# Install

Go to `https://github.com/ElsaGregoire/vocal_patterns` to see the project, manage issues,

Clone the project and install it:

```bash
git clone https://github.com/ElsaGregoire/vocal_patterns.git
cd vocal_patterns            # install and test
```
# Setup

1. Create virtualenv and install the project:
```bash
pyenv virtualenv 3.10.6 vocal_patterns
pyenv activate vocal_patterns
pip install -r requirements.txt
```

2. place the raw audio folders in the data folder

3. run `make create_csv` to create the csv file with data paths and labels
