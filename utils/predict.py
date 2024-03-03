import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

class ModelPredict:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()  
    
    def predict(self, scan):
        

