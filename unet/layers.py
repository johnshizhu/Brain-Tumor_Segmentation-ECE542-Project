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
    conv --> batch norm --> ReLU --> dropout(optional)
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

class EncoderBlock(nn.module):
    '''
    Encoder Block consists of 
    Double Conv Layer --> ReLU Activation function --> Max Pool
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        conv_kernel_size (int or tuple): Size of the convolutional kernel.
        pool_kernel_size (int or tuple): Size of the pooling kernel.
        dropout (float): Dropout probability.
        conv_stride (int or tuple): Stride of the convolution.
        conv_padding (int or tuple): Padding applied to the convolution.
        conv3d (bool): If True, 3D convolution and pooling are applied. If False, 2D convolution and pooling are applied.    
    '''
    def __init__(self, in_channels, out_channels, conv_kernel_size, pool_kernel_size, dropout, conv_stride, conv_padding, conv3d):
        super(EncoderBlock, self).__init__()
        self.doubleConv = ConvDouble(in_channels, out_channels, conv_kernel_size, True, dropout, conv_stride, conv_padding, conv3d)
        if conv3d:
            self.maxPool = nn.MaxPool3d(pool_kernel_size)
        else:
            self.maxPool = nn.MaxPool2d(pool_kernel_size)
        # Skip Connection
        self.skip_features = None

    def forward(self, x):
        post_conv_features = self.doubleConv(x)
        # Skip Connection
        self.skip_features = post_conv_features
        post_pool_features = self.maxPool(post_conv_features)
        return post_pool_features

class DecoderBlock(nn.module):
    '''
    Decoder Block consists of
    Upscaling --> Double Conv Layer

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        conv_kernel_size (int or tuple): Size of the convolutional kernel for the double convolution.
        up_kernel_size (int or tuple): Size of the transposed convolutional kernel for upscaling.
        dropout (float): Dropout probability.
        conv_stride (int or tuple): Stride of the convolution for the double convolution.
        conv_padding (int or tuple): Padding applied to the convolution for the double convolution.
        conv3d (bool): If True, 3D convolution and transpose convolution are applied. If False, 2D convolution and transpose convolution are applied.
    '''
    def __init__(self, in_channels, out_channels, conv_kernel_size, up_kernel_size, dropout, conv_stride, conv_padding, conv3d):
        super(DecoderBlock, self).__iniit__()
        self.doubleConv = ConvDouble(in_channels, out_channels, conv_kernel_size, False, dropout, conv_stride, conv_padding, conv3d)
        if conv3d:
            self.upScale = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=up_kernel_size)
        else:
            self.upScale = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=up_kernel_size)

    def forward(self, x):
        post_ups_features = self.upScale(x)
        post_conv_features = self.doubleConv(post_ups_features)
        return post_conv_features

class EncoderUNet(nn.module):
    '''
    UNet Encoder made of Encoder blocks
    '''
    def __init__(self, args):
        super(EncoderUNet, self).__init__()

    def forward(self, x):
        # FIX
        return x

class DecoderUNet(nn.module):
    '''
    UNet Decoder made of Decoder blocks
    '''
    def __init__(self, args):
        super(DecoderUNet, self).__init__()

    def forward(self, x):
        # FIX
        return x