import torch
import torch.nn as nn


def addBlock(sequence, in_channels, out_channels, kernel_size, dropout=0.1, stride=1, padding=0, conv3d=True):
    '''
    Adds a Convolutional Block, with optional batch normalization, ReLU, and dropout.
    Args:
     - sequence (string): sequence of 'c', 'b', 'r', 'd'
        - 'c': convolution 
        - 'b': batch norm
        - 'r': ReLU  
        - 'd': dropout
     - in_channels (int): Number of input channels
     - out_channels (int): Number of output channels
     - kernel_size (int or tuple): Size of the convolutional kernel
     - stride (int or tuple, optional): Stride of the convolution (default: 1)
     - padding (int or tuple, optional): Padding of the convolution (default: 0)
     - dropout (float, optional): Dropout rate (default: 0.1)
     - conv3d (boolean, optional): 
        - True: 3D Convolution (default)
        - False: 2D Convolution 
    Returns:
     - modules (list): List of PyTorch modules representing the convolutional block
    '''
    modules = []
    for i in sequence:
        # Add Convolution layer (3D or 2D)
        if i == "c":
            if conv3d:
                modules.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
            else:
                modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))

        # Add Batch norm layer    
        elif i == "b":
            if conv3d:
                modules.append(nn.BatchNorm3d(out_channels))
            else:
                modules.append(nn.BatchNorm2d(out_channels))

        # Add ReLU layer
        elif i == "r":
            modules.append(nn.ReLU(inplace=True))

        # Add Dropout
        elif i == "d":
            if conv3d:
                modules.append(nn.Dropout2d(p=dropout))
            else:
                modules.append(nn.Dropout2d(p=dropout))
    return modules

class ConvSingle(nn.Sequential):
    '''
    Single Convolutional Layer
    Args:
     - in_channels: 
     - out_channels: 
     - kernel_size: 
     - stride: 
     - padding: 
     - dropout: 
     - conv3d (boolean): 
        - True: 3D Convolution 
        - False: 2D Convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dropout, stride, padding, conv3d):
        super(ConvSingle, self).__init__()
        if dropout:
            for module in addBlock('cbrd', in_channels, out_channels, kernel_size, dropout, stride, padding, conv3d):
                self.add_module(module)
        else:
            for module in addBlock('cbr', in_channels, out_channels, kernel_size, dropout, stride, padding, conv3d):
                self.add_module(module)

class ConvDouble(nn.Sequential):
    '''
    Double Convolutional Layer
    Args:
     - in_channels: 
     - out_channels: 
     - kernel_size: 
     - endec (boolean):
        - True: encoder part of network
        - False: decoder part of network
     - stride: 
     - padding: 
     - dropout: 
     - conv3d (boolean): 
        - True: 3D Convolution 
        - False: 2D Convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size, endec, dropout, stride, padding, conv3d):
        super(ConvDouble, self).__init__()

        if endec:
            mid_channels = out_channels
        else:
            mid_channels = in_channels

        self.add_module("conv_single1", ConvSingle(in_channels, mid_channels, kernel_size, dropout, stride, padding, conv3d))
        self.add_module("conv_single2", ConvSingle(mid_channels, out_channels, kernel_size, dropout, stride, padding, conv3d))

class EncoderUNet(nn.module):
    '''
    
    '''
    def __init__(self, args):
        super(EncoderUNet, self).__init__()

    def forward(self, x):
        # FIX
        return x

class DecoderUNet(nn.module):
    def __init__(self, args):
        super(DecoderUNet, self).__init__()

    def forward(self, x):
        # FIX
        return x