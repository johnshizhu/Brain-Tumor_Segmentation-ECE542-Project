import os
import torch
import numpy as np
import nibabel as nib

class ModelPredict:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def predict(self, input_tensor):
        with torch.no_grad():  
            prediction = self.model(input_tensor)
        
        return prediction
