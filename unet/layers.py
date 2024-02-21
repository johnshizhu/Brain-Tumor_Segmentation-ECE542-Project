import torch
import torch.nn as nn


class ConvSingle(nn.Sequential):
    '''
    Single Convolutional Layer
    Args:
     - in_dim:      input channels
     - out_dim:     output channels
     - dim_type:    2D or 3D
     - conv_size:   convolution kernel size
     - stride:      stride value
     - padding:     padding value
     - dropout:     dropout rate
    '''
    def __init__(self, in_dim, out_dim, dim_type, conv_size, stride, padding, dropout):
        super(ConvSingle, self).__init__()


class ConvDouble(nn.Sequential):
    '''
    Double Convolutional Layer
    Args:
     - in_dim:      input channels
     - out_dim:     output channels
     - dim_type:    2D or 3D
     - conv_size:   convolution kernel size
     - stride:      stride value
     - padding:     padding value
     - dropout:     dropout rate
    '''
    def __init__(self, in_dim, out_dim, dim_type, conv_size, stride, padding, dropout):
        super(ConvDouble, self).__init__()



class EncoderUNet(nn.module):
    def __init__(self, args):
        super(EncoderUNet, self).__init__()


class DecoderUNet(nn.module):
    def __init__(self, args):
        super(DecoderUNet, self).__init__()