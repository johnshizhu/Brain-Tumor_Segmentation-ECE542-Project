import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
from torch.utils.data import DataLoader, TensorDataset
from unet.model import GeneralUNet
from utils.data_utils import BratsDataset3D
import numpy as np

### PARAMETERS
num_epochs = 10
lr = 0.001
batch_size = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dir = r'C:\Users\johns\OneDrive\Desktop\Datasets\ECE-542\brain-tumor-segmentation(nii)\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'

train_dataset = BratsDataset3D(train_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model setup
model = GeneralUNet(in_channels=1,  # Adjust based on your dataset's specifics
                    conv_kernel_size=3,
                    pool_kernel_size=2,
                    up_kernel_size=2,
                    dropout=0.1,
                    conv_stride=1,
                    conv_padding=1,
                    conv3d=True,
                    size=2,  # Adjust the number of layers in the UNet
                    complex=4)  # Adjust the complexity or number of initial features


# Loss and optimizer
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=lr)


model = model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        print(f'forward pass complete')
        loss = criterion(outputs, labels)
        print(f'loss is: {loss}')

        # Backward and optimize
        optimizer.zero_grad()
        print(f'backprop')
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

print("Training complete!")
