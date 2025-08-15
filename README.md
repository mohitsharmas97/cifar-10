# CIFAR-10 Image Classification with PyTorch

This project details the implementation of a Convolutional Neural Network (CNN) using PyTorch for classifying images from the well-known CIFAR-10 dataset. The model demonstrates the process of building, training, and evaluating a deep learning model for image classification, achieving a test accuracy of approximately 81.97%.


https://www.kaggle.com/datasets/ayush1220/cifar10


The model's performance was evaluated using accuracy as the primary metric.

-   **Training Accuracy:** 86.79%
-   **Test Accuracy:** 81.97%


***

## About the Dataset

The **CIFAR-10 dataset** is a staple for computer vision tasks and consists of 60,000 32x32 color images across 10 classes. The dataset is pre-divided into 50,000 training images and 10,000 test images.

The 10 classes include:
- airplane 
- automobile 
- bird 
- cat 
- deer 
- dog 
- frog 
- horse 
- ship 
- truck 

To enhance the model's ability to generalize, the training data undergoes **augmentation**, including random horizontal flips and rotations. All images are normalized before being fed into the network.

***

##  Model Architecture

A **Convolutional Neural Network (CNN)** is employed for this classification task. The architecture is composed of two main parts: a feature extractor and a classifier.

### Feature Extractor:
1.  **Convolutional Layer 1**: 32 filters, 3x3 kernel, followed by ReLU activation, and Batch Normalization.
2.  **Max Pooling Layer 1**: 2x2 kernel.
3.  **Convolutional Layer 2**: 64 filters, 3x3 kernel, followed by ReLU activation, and Batch Normalization.
4.  **Max Pooling Layer 2**: 2x2 kernel.
5.  **Convolutional Layer 3**: 128 filters, 3x3 kernel, followed by ReLU activation, and Batch Normalization.
6.  **Max Pooling Layer 3**: 2x2 kernel.

### Classifier:
1.  **Flatten Layer**
2.  **Linear Layer**: 256 output features with ReLU activation.
3.  **Dropout**: A dropout rate of 0.5 is used for regularization.
4.  **Linear Layer**: 128 output features with ReLU activation.
5.  **Dropout**: Another dropout layer with a rate of 0.5.
6.  **Output Layer**: A final linear layer with 10 output features, corresponding to the 10 classes.

***

##  Getting Started

To get this project up and running on your local machine, follow these steps.

### Prerequisites

Ensure you have Python installed. The project relies on the following libraries:
- `torch`
- `torchvision`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `numpy`
- `pandas`
- `kagglehub`

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    ```
2.  **Install the required packages:**
    ```sh
    pip install torch torchvision scikit-learn matplotlib seaborn numpy pandas kagglehub
    ```

***

## Usage

1.  Run the Jupyter Notebook `Cifar_10_by_pytorch.ipynb`.
2.  The notebook will first download the CIFAR-10 dataset.
3.  It will then proceed to train the CNN model for 40 epochs.
4.  After training, it will evaluate the model's performance on both the training and test datasets.
5.  The trained model weights will be saved to a file named `cifar10_model.pth`.

***

##  Results & Evaluation

The model was trained and evaluated with the following results:

-   **Training Accuracy:** 86.79%
-   **Test Accuracy:** 81.97%

### Confusion Matrix

The confusion matrix below visualizes the performance of the model on the test set, showing the number of correct and incorrect predictions for each class.

<img width="788" height="701" alt="image" src="https://github.com/user-attachments/assets/daebb0a3-c7ce-4939-8bea-2836ad8e19be" />


