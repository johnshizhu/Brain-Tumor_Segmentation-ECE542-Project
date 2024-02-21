import torch
import torch.nn as nn

class GeneralUNet(nn.Module):
    '''
    General class for UNet Structure
    '''
    def __init__(self, args):
        super(GeneralUNet, self).__init__()


    def forward(self, x):
        return



class UNet3D(GeneralUNet):
    def __init__(self, moreargs):
        super(UNet3D, self).__init__()