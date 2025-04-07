import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io
import glob
from PIL import Image
import utils
import json

"""
Vilkas Door - Automated Dog Recognition System

This module implements a computer vision system that recognizes a specific dog (Vilkas) 
using transfer learning with a pre-trained VGG16 model. The system is designed to be 
used in an automated dog door that only opens for Vilkas.

The implementation uses PyTorch and follows these key steps:
1. Loads a pre-trained VGG16 model
2. Adds a custom classification layer
3. Trains the custom layer while keeping VGG16 layers frozen
4. Fine-tunes the entire model with a very small learning rate

Requirements:
    - torch
    - torchvision
    - PIL
    - matplotlib
"""

# The end goal of the project is to build an automated dog door that only recognizes and opens when one specific dog, Vilkas approaches.
# This initial routine is a proof of concept for a image classifer based on a pre-trained model(ImageNet VGG16). 
# ImageNet models have learned to detect animals, including dogs, it is especially well suited for this transfer learning task of detecting Vilkas.


# To the last layer of the ImageNet model (which has 1000 clasess), an additonal layer providing a single classification Vilkas / No Vilkas 
# will be added. 
# The base model will be frozen, and only the new layer will be trained initially so as to preserve the learning achieved from training 
# on the ImageNet dataset and not overfit. The entire model will then be unfrozen and fine-tuned with a small learning rate.


# This version is to run on a Metal Performance Shader (MPS) GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.is_available()


# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
#device = "cpu"
print(f"Using device: {device}")

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

# load the VGG16 network *pre-trained* on the ImageNet dataset
weights = VGG16_Weights.DEFAULT
vgg_model = vgg16(weights=weights)

# Load the model to the GPU
# vgg_model.to(device)


#Freeze the pre-trained layers

# Verify the state of VGG layers,  by looping through the model parameters.
#vgg_model.requires_grad_(True)
#print("VGG16 Unfrozen")
#for idx, param in enumerate(vgg_model.parameters()):
#    print(idx, param.requires_grad)

vgg_model.requires_grad_(False)
print("VGG16 Frozen")


# Input dimenisons, use pre-trained model weights which supply transform methods

pre_trans = weights.transforms()
#pre_trans# Load the image

# The code is equivalent to the following:

#IMG_WIDTH, IMG_HEIGHT = (224, 224)

#pre_trans = transforms.Compose([
#    transforms.ToDtype(torch.float32, scale=True), # Converts [0, 255] to [0, 1]
#    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
#    transforms.Normalize(
#        mean=[0.485, 0.456, 0.406],
#        std=[0.229, 0.224, 0.225],
#    ),
#    transforms.CenterCrop(224)
#])

#Adding new layers to the model

#We can now add the new trainable layers to the pre-trained model. They will take the 
# features from the pre-trained layers and turn them into predictions on the new dataset. We 
# will add two layers to the model. In a previous lesson, we created our own custom module. 
# A transfer learning module works in the exact same way. We can use is a layer in a 
# Define a Sequential Model with baseline vgg_model followed by a Linear layer connecting all 1000 of VGG16's outputs to 1 neuron.

N_CLASSES = 1

my_model = nn.Sequential(
    vgg_model,
    nn.Linear(1000, N_CLASSES)
)

my_model.to(device)

# Use binary crossentropy and binary accuracy as this is a binary classification problem (Vilkas or not)

#By setting from_logits=True we inform the loss function that the output values are 
# not normalized (e.g. with softmax).

loss_function = nn.BCEWithLogitsLoss()
optimizer = Adam(my_model.parameters())
my_model = my_model.to(device)


# Read image files directly and infer the label based on the filepath.
DATA_LABELS = ["vilkas", "not_vilkas"] 
    
class MyDataset(Dataset):
    """
    Custom dataset class for loading and preprocessing dog images.
    
    The dataset expects images to be organized in directories named after their classes
    (vilkas/ and not_vilkas/).

    Args:
        data_dir (str): Root directory containing class-specific subdirectories of images

    Returns:
        tuple: (preprocessed_image, label)
    """
    def __init__(self, data_dir):
        self.imgs = []
        self.labels = []
        
        for l_idx, label in enumerate(DATA_LABELS):
            data_paths = glob.glob(data_dir + label + '/*.jpg', recursive=True)
            for path in data_paths:
                img = Image.open(path)
                self.imgs.append(pre_trans(img).to(device))
                self.labels.append(torch.tensor(l_idx).to(device).float())


    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)

#Data Loaders
n = 32

train_path = "data/train/"
train_data = MyDataset(train_path)
train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)

valid_path = "data/validate/"
valid_data = MyDataset(valid_path)
valid_loader = DataLoader(valid_data, batch_size=n)
valid_N = len(valid_loader.dataset)

# Augment the class data to improve recognition chances
IMG_WIDTH, IMG_HEIGHT = (224, 224)

random_trans = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.8, 1), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2)
])


# get_batch_accuracy function will use Binary Cross Entropy as a loss function. 
# The output could have been through the sigmoid function, but this more efficient in this special case. 


def get_batch_accuracy(output, y, N):
    """
    Calculate binary classification accuracy for a batch.

    Args:
        output (torch.Tensor): Model predictions
        y (torch.Tensor): Ground truth labels
        N (int): Total number of samples

    Returns:
        float: Accuracy as a fraction of correct predictions
    """
    zero_tensor = torch.tensor([0]).to(device)
    pred = torch.gt(output, zero_tensor)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

# Print the last set of gradients to show that only our newly added layers are learning.

def train(model, check_grad=False):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model to train
        check_grad (bool): If True, prints gradients after training
    """
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = torch.squeeze(model(random_trans(x)))
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    if check_grad:
        print('Last Gradient:')
        for param in model.parameters():
            print(param.grad)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


# Display a sample of the model's gradients. This voluminous because VGG16 final layer has 1000 classes, 1000 weights connected to the single neuron in the next layer. 

#train(my_model, check_grad=True)

# Validate the model 

def validate(model):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model to evaluate
    """
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = torch.squeeze(model(x))

            loss += loss_function(output, y.float()).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

epochs = 20

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train(my_model, check_grad=False)
    validate(my_model)



# We can now fine-tune the model. Fine-tuning is the process of updating the pre-trained layers 
# of the model. This is done by unfreezing the layers and training the model on a new dataset. 
# Use the same training loop as before, but we will update the pre-trained layers.
# This will cause the base pre-trained layers to take very small steps and adjust slightly, improving the model by a small amount.
# VGG16 is a relatively large model, so the small learning rate will also prevent overfitting.
# Note that it is important to only do this step after the model with frozen layers has been fully trained.
# The untrained linear layer that we added to the model earlier was randomly initialized.
# This means it needed to be updated quite a lot to correctly classify the images.
# Through the process of backpropagation, large initial updates in the last layers would have caused potentially large updates in the pre-trained layers as well.
# These updates would have destroyed those important pre-trained features.
# However, now that those final layers are trained and have converged, any updates to the model as a whole will be much smaller
# (especially with a very small learning rate) and will not destroy the features of the earlier layers.

# Unfreeze the pre-trained layers, and then fine tuning the model:

# Unfreeze the base model
vgg_model.requires_grad_(True)
optimizer = Adam(my_model.parameters(), lr=.000001)


# Train for only a few epochs. VGG16 is a large model, it can overfit when excessive training.
 
epochs = 3

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train(my_model, check_grad=False)
    validate(my_model)


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    """
    Display an image using matplotlib.

    Args:
        image_path (str): Path to the image file
    """
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.show()

def make_prediction(file_path):
    """
    Make a prediction on a single image.

    Args:
        file_path (str): Path to the image file

    Returns:
        float: Prediction score (negative values indicate Vilkas)
    """
    show_image(file_path)
    image = Image.open(file_path)
    image = pre_trans(image).to(device)
    image = image.unsqueeze(0)
    output = my_model(image)
    prediction = output.item()
    return prediction


# vilkas is in the -1 class

def vilkas_doggy_door(image_path):
    """
    Simulate the dog door's decision making process.
    
    Makes a prediction on the input image and prints whether the door should
    open for the detected dog.

    Args:
        image_path (str): Path to the image file to analyze
    """
    pred = make_prediction(image_path)
    if pred <0:
        print("It's Vilkas! Let him in!")
    else:
        print("That's not Vilkas! Stay out!")

# Anecdotally test the model
vilkas_doggy_door('data/validate/not_vilkas/133.jpg')
vilkas_doggy_door('data/validate/vilkas/IMG_0027.jpg')
vilkas_doggy_door('data/validate/vilkas/IMG_3291.jpg')


