
# Lidar Point Cloud Semantic Segmentation using Range Images

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-%E2%89%A53.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-%E2%89%A52.1-brightgreen.svg)](https://pytorch.org/)


## Overview

This repository contains the code for performing semantic segmentation of lidar point clouds - specifically on SemanticKitti dataset - by leveraging range image representations. This approach transforms sparse 3D point clouds into dense 2D images, enabling the application of efficient 2D convolutional neural networks for pixel-wise classification, which corresponds to the semantic labels of the original 3D points. Previous works on SemanticKitti dataset revealed that the lack of training data significantly impact the generalization of model. While traditional augmentation techniaues has no to little effect on Range Images, this code implements an idea of augmenting rare-object classes along with well-known CutMix augmentation technique that notably improves the performance, **from about 46% to 51.5% IoU score** (without post-processing) on the validation set. The idea is to extract rare-objects from the training dataset and store them into a database; during training an augmentation module randomly selects different objects from the databse and augments them to the Range Images.

The code barrows implementation of [range image creation](https://github.com/robot-learning-freiburg/EfficientLPS/) and [post-processing KNN](https://github.com/PRBonn/lidar-bonnetal) from previous published works. The code also utilizes the well-known [Segmentation Models Pytorch ](https://github.com/qubvel-org/segmentation_models.pytorch/releases) that enables the user to select from a wide veriety of segmentation architectures (such as, Unet, Unet++, DeepLabV3+, SegFormer, etc.) 

**Key Features:**

* **Data Augmentation with Rare-object Injection:** Implements a method to generate new data by injecting rare-object classes.
* **Modular Network Architecture:** Designed with flexibility in mind, allowing for the integration of various 2D backbone networks (e.g., Darknet53).
* **Customizable Data Loading:** Supports custom lidar datasets with provided tools for data preparation and loading.
* **Common Loss Functions:** Includes implementations of Dice Loss and Jaccard Loss, commonly used for segmentation tasks.

## Getting Started

In order to setup this repo locally take the following steps.

### Prerequisites

Ensure you have the following installed:

* Python (>= 3.10)
* PyTorch (>= 2.1)
* CUDA (if you intend to use GPU acceleration)
* Other required Python packages (install via `pip install -r requirements.txt`)

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

**[Instructions on how to prepare your lidar dataset will be added here.]**

Refer to the `data` directory for data loading and dataset-specific configurations.

### Configuration

The project utilizes YAML files for configuration. You can find example configuration files in the `configs` directory.

* `data_config.yaml`: Contains settings related to the dataset, such as file paths, class mappings, and data augmentation parameters.
* `darknet53.yaml`: An example configuration for a Darknet53-based segmentation model.

**[More details about the configuration options will be added here.]**

### Training

To train a model, use the `train.py` script with the desired configuration file:

```bash
python train.py --config configs/your_config.yaml
