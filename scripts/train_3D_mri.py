import sys
sys.path.insert(0, '../')
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
from torch.utils.data import DataLoader, TensorDataset
from unet.model import GeneralUNet, UNet3D
from utils.data_utils import BratsDataset3D
import numpy as np

'''
Training Loop Script

Trains a 3DUnet using the default channel and scaling settings in this library

arg0 = number of epochs (int)
arg1 = batch size (int)
arg2 = learning rate (float)
arg3 = directory of data (file path)
arg4 = number of in_channels (int)
arg5 = size: depth of unet(depth) (int)
arg6 = complex: inital value to scale to (int)
'''

### PARAMETERS
num_epochs = sys.argv[0]
batch_size = sys.argv[1]
lr         = sys.argv[2]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dir  = sys.argv[3]

train_dataset = BratsDataset3D(train_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model setup 3D Unet using defaults
model = UNet3D(in_channels  = sys.argv[4], 
               size         = sys.argv[5], 
               complex      = sys.argv[6])  

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
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

print("Training complete!")
