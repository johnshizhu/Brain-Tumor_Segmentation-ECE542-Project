import torch
import torch.nn as nn
from layers import EncoderBlock, DecoderBlock, EncoderUNet, DecoderUNet

class GeneralUNet(nn.Module):
    '''
    General class for UNet Structure
    '''
    def __init__(self, in_channels, out_channels, conv_kernel_size, pool_kernel_size, up_kernel_size, dropout, conv_stride, conv_padding, conv3d, size):
        super(GeneralUNet, self).__init__()
        self.encoder_series = EncoderUNet(in_channels, out_channels, conv_kernel_size, pool_kernel_size, dropout, conv_stride, conv_padding, conv3d, size)
        self.decoder_series = DecoderUNet(in_channels, out_channels, conv_kernel_size, up_kernel_size, dropout, conv_stride, conv_padding, conv3d)

    def forward(self, x):
        encoder_features = self.encoder(x)
        output_features = self.decoder(encoder_features)
        return output_features

class UNet3D(GeneralUNet):
    def __init__(self, moreargs):
        super(UNet3D, self).__init__()