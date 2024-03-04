import os
import torch
import numpy as np
import nibabel as nib

class ModelPredict:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
    
    def get_scan(self, scan_folder, scan_type, index):
        scan_path = os.path.join(scan_folder, f'BraTS20_Training_{index+1:03d}_{scan_type}.nii')
        scan = nib.load(scan_path)
        scan_data = scan.get_fdata()
        
        return scan_data
    
    def predict(self, input_tensor):
        with torch.no_grad():  
            prediction = self.model(input_tensor)
        
        return prediction
