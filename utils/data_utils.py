import os
import torch
import nibabel as nib
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
    
    def __getitem__(self, idx, type='flair'):
        '''
        Returns a list containing an data point (MRI scan) and its corresponding label (segmentation)
        Args:
            idx     (int): The index valuye
            type    (string): The file type
                - 'flair'
                - 't1'
                - 't1ce'
                - 't2'
        Output:
         - image: (240, 240, 155) MRI slices
         - label: (240, 240, 155) Volumetric Segmentation labels
        '''
        # Raise error if type is seg
        if type == 'seg':
            raise ValueError("Invalid value for 'type': 'seg' is reserved for segmentation labels file type")
        
        # Construct file paths for sample
        scan_folder = self.samples[idx]
        image_path = os.path.join(scan_folder, f'BraTS20_Training_{idx+1:03d}_{type}.nii')
        seg_path = os.path.join(scan_folder, f'BraTS20_Training_{idx+1:03d}_seg.nii')

        # Load data
        image = nib.load(image_path).get_fdata()
        label = nib.load(seg_path).get_fdata()

        if self.transform:
            sample = self.transform(sample)

        return image, label
    
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
