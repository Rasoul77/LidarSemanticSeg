
# Lidar Point Cloud Semantic Segmentation using Range Images

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-%E2%89%A53.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-%E2%89%A52.1-brightgreen.svg)](https://pytorch.org/)


## Overview

This repository contains the code for performing semantic segmentation of lidar point clouds - specifically on SemanticKitti dataset - by leveraging range image representations. This approach transforms sparse 3D point clouds into dense 2D images, enabling the application of efficient 2D convolutional neural networks for pixel-wise classification, which corresponds to the semantic labels of the original 3D points. Previous works on SemanticKitti dataset revealed that the lack of training data significantly impact the generalization of model. While traditional augmentation techniaues has no to little effect on Range Images, this code implements an idea of augmenting rare-object classes along with well-known CutMix augmentation technique that notably improves the performance **by 5.5 points of percentage of IoU score** on the validation set. The idea is to extract rare-objects from the training dataset and store them into a database; during training an augmentation module randomly selects different objects from the databse and augments them to the Range Images.

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
    git clone [https://github.com/Rasoul77/LidarSemanticSeg.git](https://github.com/Rasoul77/LidarSemanticSeg.git)
    
    cd LidarSemanticSeg
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

Download [SemanticKitti](https://www.semantic-kitti.org/dataset.html) dataset locally, then from this repo, open `code/configs/data_config.yaml` and update `seq_dir`'s value with the full path to the `sequences` folder of the SemanticKitti dataset,
```
seq_dir: "full/path/to/SemanticKITTI/dataset/sequences"
```

#### Run Statistics Calculations
In order to standardize the input data, we need to calcualte `mean` and `std` of the training set. The script `code/tools/calc_trainset_stat.py` implements a memory efficient algorithm to calculate the statistics of the training set data. You need to run this script so that the calculated statistics will be written to the `data` folder and be used during training and evaluation.
```
cd code
python3 -m tools.calc_trainset_stat
```

#### Run Rare-object Database Creator
In order to generate new data using rare-object injection augmentation, we need to create a database of such objects first. The script `code/tools/extract_rare_objects.py` implements the algorithm, and it writes the extracted objects to the folder `code/data/objects_db`.
```
cd code
python3 -m tools.extract_rare_objects
```

### Configuration

The project utilizes YAML files for configuration. You can find example configuration files in the `code/configs` directory.

* `data_config.yaml`: Contains settings related to the dataset, such as file paths, class mappings, and data augmentation parameters.
* `darknet53.yaml`: An example configuration for a Darknet53-based segmentation model.

### Training

To train a model, use the `train.py` script with the desired configuration file and optional `wandb` logging, for example:

```bash
cd code
python3 train.py --config_name darknet53.yaml --use-wandb --name unet_darknet53_40epochs
```

You should see a progress reporting at the end of each epoch like this:

```
...
Epoch35/40 | Train > Epoch Loss: 0.6682 Time: 03:23 | Eval > Epoch Loss: 0.7265 SCORE: 51.34% Time: 00:40
Per Class Score:
car            82.950%
bicycle        36.634%
motorcycle     39.490%
truck          60.984%
other-vehicle  23.982%
person         55.955%
bicyclist      61.259%
motorcyclist   0.150%
road           87.059%
parking        38.937%
sidewalk       75.297%
other-ground   1.081%
building       77.224%
fence          39.945%
vegetation     74.949%
trunk          58.298%
terrain        63.558%
pole           59.270%
traffic-sign   38.455%
...
```
