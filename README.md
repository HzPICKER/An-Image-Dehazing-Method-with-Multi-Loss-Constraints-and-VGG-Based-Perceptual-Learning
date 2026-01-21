An Image Dehazing Method with Multi-Loss Constraints and VGG-Based Perceptual Learning

This repository provides the official implementation and dataset organization for an image dehazing framework based on deep learning.
The proposed method aims to restore clear images from single hazy inputs by learning a direct mapping between hazy and clean image pairs under multiple loss constraints, including perceptual supervision.

The code is designed to support both model training and inference, and can be used for research and academic purposes.

1. Project Overview

Single-image dehazing is a challenging low-level vision task due to the ill-posed nature of haze degradation, which depends on scene depth, illumination, and atmospheric conditions.
This project implements a learning-based image dehazing approach that leverages pixel-level fidelity and perceptual consistency to improve visual quality and structural preservation.

The repository includes:

A reference dataset organization for supervised image dehazing

A trained generator model for inference

A Python-based application for single-image and batch dehazing

2. Directory Structure
.
├── dehaze
│   ├── clear_images        # Ground-truth clear images
│   ├── haze                # Corresponding hazy images
│   └── README.md           # Dataset description
│
├── filtered-dehaze
│   └── filtered
│       ├── train           # Filtered training samples
│       └── test            # Filtered testing samples
│
├── main.py                 # Model training and related components
├── keras01.py              # Inference application
└── README.md               # Project documentation

3. Dataset Organization
3.1 Paired Dehazing Dataset (dehaze/)

The dehaze/ directory follows a paired image setting, which is commonly adopted in supervised image dehazing research.

haze/
Contains hazy input images.

clear_images/
Contains the corresponding ground-truth clear images.

Each hazy image should have a one-to-one correspondence with a clear image.

3.2 Filtered Dataset (filtered-dehaze/filtered/)

The filtered-dehaze/filtered/ directory contains refined subsets used for controlled experiments:

train/
Filtered samples used for model training.

test/
Filtered samples used for quantitative and qualitative evaluation.

This separation allows reproducible training and testing under consistent conditions.

4. Method Description

The implemented model learns an end-to-end mapping from hazy images to clear images using a deep convolutional neural network.
To improve restoration quality, the training process incorporates multiple complementary loss functions, including:

Pixel-wise reconstruction loss

Structural similarity constraints

VGG-based perceptual loss to enhance high-level feature consistency

The final trained generator is saved in Keras format (.keras) and can be directly used for inference without additional training.

5. Requirements

The implementation is based on Python and TensorFlow.
The main dependencies include:

Python ≥ 3.8

TensorFlow

OpenCV

NumPy

Matplotlib

Pillow

Example installation:

pip install tensorflow opencv-python numpy matplotlib pillow

6. Inference Usage

The script keras01.py provides a standalone interface for image dehazing.

6.1 Single Image Dehazing
python keras01.py hazy.jpg


The dehazed result will be saved as:

hazy_dehazed.jpg


Specify an output path:

python keras01.py hazy.jpg dehazed.jpg

6.2 Batch Image Dehazing
python keras01.py --batch input_folder/ output_folder/


All supported image formats in the input folder will be processed automatically.

6.3 Using a Custom Model
python keras01.py --model best_generator.keras hazy.jpg

6.4 Visual Comparison

To generate a side-by-side comparison between the hazy input and the dehazed result:

python keras01.py --compare hazy.jpg dehazed.jpg comparison.jpg

6.5 Model Test

A simple functionality test can be performed using:

python keras01.py --test

7. Implementation Notes

All input images are resized to 256 × 256 before inference.

Output images are resized back to the original resolution.

Pixel values are normalized to the range [0, 1] during model processing.

8. Citation

If you use this repository or its codebase in your research, please cite the corresponding paper:

@article{dehazing_multiloss_vgg,
  title   = {An Image Dehazing Method with Multi-Loss Constraints and VGG-Based Perceptual Learning},
  author  = {},
  journal = {},
  year    = {}
}

9. License

This project is intended for academic and research use only.
For commercial usage, please contact the authors.