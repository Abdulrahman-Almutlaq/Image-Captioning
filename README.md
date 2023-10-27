# Image Captioning Project

This project focuses on generating descriptive captions for images using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The goal is to automatically generate textual descriptions for a given image. This project consists of several Jupyter notebooks for different stages of the process.

## Project Structure

- **0_Dataset.ipynb**:
  - Introduction to the COCO dataset.
  - Initializing the COCO API.
  - Plotting a sample image and its captions.
  - Preparing for the project.


- **1_Preliminaries.ipynb**:
  - Introduction to the project and dataset.
  - Data preprocessing steps.
  - Designing a CNN-RNN model for image captioning.
  - Implementing the encoder and decoder.

- **2_Training.ipynb**:
  - Training the image captioning model.
  - Exploring data loaders and preprocessing.
  - Experimenting with the CNN encoder.
  - Implementing the RNN decoder.

- **3_Inference.ipynb**:
  - Using the trained model to generate captions for test dataset images.
  - Submission notebook for grading.

- **data_loader.py**:
  - Data loading and preprocessing utilities.

- **model.py**:
  - Definitions of CNN and RNN models.

- **vocabulary.py**:
  - Vocabulary creation and word-to-index mapping.

- **vocab.pkl**:
  - Vocabulary file to map words to indices.
  

## Getting Started

To run the project, follow these steps:

1. Clone the repository.
2. Run `pip install -r requirements.txt` in the project directory.
3. Open the Jupyter notebooks to explore and run the code.

In the Jupyter notebooks, you'll find detailed explanations, instructions, and code for each part of the project. If you want to witness the magic of image captioning live, the inference notebook `3_Inference.ipynb` is the destination.

For additional information about the dataset, model architecture, and specific implementation details, please refer to the respective Jupyter notebooks.