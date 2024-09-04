

---

# Cat-Dog Classifier using CNN

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model is trained on a dataset of labeled images and can accurately predict whether a given image contains a cat or a dog.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Cat-Dog Classifier is a deep learning project that utilizes Convolutional Neural Networks (CNN) to distinguish between images of cats and dogs. CNNs are particularly well-suited for image classification tasks due to their ability to capture spatial hierarchies in images.

## Dataset

The model is trained on the [Kaggle Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data), which contains 25,000 images of cats and dogs. The dataset is split into training, validation, and test sets.

- **Training Set:** 20,000 images
- **Validation Set:** 2,500 images
- **Test Set:** 2,500 images

## Model Architecture

The CNN model is built using the following layers:

1. **Convolutional Layer:** Extracts features from the input images using filters.
2. **Max Pooling Layer:** Reduces the spatial dimensions of the feature maps.
3. **Flattening Layer:** Converts the 2D matrices into a 1D vector.
4. **Fully Connected Layer:** Applies a neural network for classification.
5. **Output Layer:** Uses a sigmoid activation function to output the probability of the image being a cat or a dog.

The architecture of the CNN is as follows:

```plaintext
Input Layer -> Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D -> Flatten -> Dense -> Output Layer
```

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ShawneilRodrigues/cat-dog-classifier.git
    ```
2. **Navigate to the project directory:**
    ```bash
    cd cat-dog-classifier
    ```
3. **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
4. **Activate the virtual environment:**
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
5. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model and classify images, follow these steps:

1. **Prepare the dataset:**
   - Download the dataset from Kaggle.
   - Extract the dataset into the `data/` directory.

2. **Train the model:**
    ```bash
    python train.py
    ```

3. **Evaluate the model:**
    ```bash
    python evaluate.py
    ```

4. **Make predictions:**
    ```bash
    python predict.py --image_path path_to_image.jpg
    ```

## Results

After training the model, it achieves an accuracy of approximately **XX%** on the test set. The model can accurately classify most images of cats and dogs.

Here are some sample predictions:

- Image 1: **Dog** (Confidence: 0.98)
- Image 2: **Cat** (Confidence: 0.95)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

