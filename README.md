# Vilkas Door - Automated Dog Recognition System 🐕

## Overview
Vilkas Door is a computer vision system that uses deep learning to recognize a specific dog (Vilkas) and could be integrated into an automated dog door system. The system employs transfer learning with a pre-trained VGG16 model to achieve high accuracy in dog recognition while requiring minimal training data.

## Features
- Custom dog recognition using transfer learning
- Built on pre-trained VGG16 architecture
- Real-time image classification
- Support for GPU acceleration (CUDA and MPS)
- Data augmentation for improved model robustness
- Binary classification (Vilkas/Not Vilkas)

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- Pillow (PIL)
- matplotlib

## Installation

bash
Clone the repository
git clone https://github.com/yourusername/vilkas-door.git
Install dependencies
pip install torch torchvision pillow matplotlib

Structure <br>
```
pet_detector/
├── data/
│   ├── train/
│   │   ├── vilkas/
│   │   └── not_vilkas/
│   └── validate/
│       ├── vilkas/
│       └── not_vilkas/
├── vilkas_door.py
└── README.md
```



## Usage
1. Organize your training data in the following structure:
Where your #1 favorite dog takes the place of Vilkas.  Subsittue your dog's name and images for the training and validation data.  
A minimum of about 30 images are suggested for training and at least 10-20 for validation testing. 
.  
   - `data/train/vilkas/` - Training images of Vilkas
   - `data/train/not_vilkas/` - Training images of other dogs
   - `data/validate/vilkas/` - Validation images of Vilkas
   - `data/validate/not_vilkas/` - Validation images of other dogs

2. Run the script:
python vilkas_door.py
:
The script will go through a fix number of training epocs for the new classification layer, a very small number of epocs for model fine-tuning.
Classification will then be conducted on the provided samples configured in the script. 


## Model Architecture
The system uses a two-stage training approach:
1. **Initial Training**: Trains only the custom classification layer while keeping the pre-trained VGG16 layers frozen
2. **Fine-tuning**: Carefully updates the entire model with a very small learning rate to optimize performance

## Data Augmentation
The system employs several data augmentation techniques to improve model robustness:
- Random rotation (±25 degrees)
- Random resized cropping
- Random horizontal flipping
- Color jittering (brightness, contrast, saturation, hue)

## Acknowledgments
- The VGG16 model architecture and pre-trained weights are from the torchvision library
- This project was inspired by the need for an automated dog door system that can recognize specific pets
