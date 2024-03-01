import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

class BratsDataset3D(Dataset):
    def __init__(self, root_dir, transform=None):
        '''
        Pytorch Dataset for full set of MRI data 
        Args:
            root_dir (string): Directory with folder containing training data
            transform (callable, optional): Optional transform to apply on a sample
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [os.path.join(root_dir, o) for o in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, o))]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        '''
        Returns a list containing an data point (MRI scan) and its corresponding label (segmentation)
        Args:
            idx     (int): The index valuye
        Output:
         - image: (1, 4, 240, 240, 155) MRI Scan data
         - label: (1, 1, 240, 240, 155) Volumetric Segmentation labels
        '''
        # Raise error if type is seg
        if type == 'seg':
            raise ValueError("Invalid value for 'type': 'seg' is reserved for segmentation labels file type")
        
        # Construct file paths for sample
        scan_folder = self.samples[idx]
        flair_path  = os.path.join(scan_folder, f'BraTS20_Training_{idx+1:03d}_flair.nii')
        t1_path     = os.path.join(scan_folder, f'BraTS20_Training_{idx+1:03d}_t1.nii')
        t1ce_path   = os.path.join(scan_folder, f'BraTS20_Training_{idx+1:03d}_t1ce.nii')
        t2_path     = os.path.join(scan_folder, f'BraTS20_Training_{idx+1:03d}_t2.nii')
        seg_path    = os.path.join(scan_folder, f'BraTS20_Training_{idx+1:03d}_seg.nii')

        # Load data
        flair   = nib.load(flair_path).get_fdata()
        t1      = nib.load(t1_path).get_fdata()
        t1ce    = nib.load(t1ce_path).get_fdata()
        t2      = nib.load(t2_path).get_fdata()
        label   = nib.load(seg_path).get_fdata()

        flair_tensor = torch.tensor(flair).float()
        t1_tensor    = torch.tensor(t1).float()
        t1ce_tensor  = torch.tensor(t1ce).float()
        t2_tensor    = torch.tensor(t2).float()
        label_tensor = torch.tensor(label).float()

        # Normalize Scan information
        flair_norm = (flair_tensor - flair_tensor.min()) / (flair_tensor.max() - flair_tensor.min())
        t1_norm = (t1_tensor - t1_tensor.min()) / (t1_tensor.max() - t1_tensor.min())
        t1ce_norm = (t1ce_tensor - t1ce_tensor.min()) / (t1ce_tensor.max() - t1ce_tensor.min())
        t2_norm = (t2_tensor - t2_tensor.min()) / (t2_tensor.max() - t2_tensor.min())
        
        # Correct label information
        label_tensor[label_tensor == 2] = 0
        label_tensor[label_tensor == 4] = 1

        if self.transform:
            sample = self.transform(sample)

        image_tensor = torch.stack([flair_norm, t1_norm, t1ce_norm, t2_norm], dim=0)

        label_tensor = label_tensor.unsqueeze(0)

        return image_tensor.float(), label_tensor.float()
    
    def modify_seg_labels(self):
        for subdir, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("_seg.nii"):
                    file_path = os.path.join(subdir, file)
                    seg_img = nib.load(file_path)
                    seg_data = seg_img.get_fdata()
                    
                    modified_seg_data = np.where((seg_data == 1) | (seg_data == 4), 1, 0)
                    mod_seg_path = file_path.replace("_seg.nii", "_seg_mod.nii")
                    modified_seg_img = nib.Nifti1Image(modified_seg_data, seg_img.affine, seg_img.header)
                    nib.save(modified_seg_img, mod_seg_path)
    
class BratsDataset2D(Dataset):
    def __init__(self, root_dir):
        '''
        Pytorch Dataset for single slice MRI images
        Args:
            root_dir (string): Directory with folder containing training data
            transform (callable, optional): Optional transform to apply on a sample
        '''
        self.root_dir = root_dir
        self.samples = None # FIXHERE should be file paths fro each individual image

        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self):
            ### FIX THIS
            image = None
            label = None
            
            return image, label
