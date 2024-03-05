import torch
import torch.nn as nn
from .layers import EncoderBlock, DecoderBlock, EncoderUNet, DecoderUNet, Bottleneck

class GeneralUNet(nn.Module):
    '''
    Generalized U-Net architecture supporting both 2D and 3D convolutions, designed for segmentation tasks.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels in the output segmentation map.
        conv_kernel_size (int or tuple): Size of the kernel for the convolutional layers.
        pool_kernel_size (int or tuple): Size of the kernel for the pooling layers in the encoder.
        up_kernel_size (int or tuple): Size of the kernel for the transposed convolutional layers in the decoder.
        dropout (float, optional): Dropout rate used in bottleneck and decoder layers. Default: None (no dropout).
        conv_stride (int or tuple): Stride for the convolutional operations.
        conv_padding (int or tuple): Padding for the convolutional operations.
        conv3d (bool): Flag indicating whether to use 3D convolutions (True) or 2D convolutions (False).
        size (int): Number of layers in the encoder/decoder.
    '''
    def __init__(self, in_channels, conv_kernel_size, pool_kernel_size, up_kernel_size, dropout, conv_stride, conv_padding, conv3d, size, complex):
        super(GeneralUNet, self).__init__()
        self.encoder_series = EncoderUNet(in_channels, conv_kernel_size, pool_kernel_size, dropout, conv_stride, conv_padding, conv3d, size, complex)
        self.bottleneck     = Bottleneck(complex * (2 ** (size-1)), conv_kernel_size, conv_stride, conv_padding, dropout, conv3d)
        self.decoder_series = DecoderUNet(complex * (2 ** (size)), conv_kernel_size, up_kernel_size, dropout, conv_stride, conv_padding, conv3d, size, complex)
        if conv3d:
            self.last_conv = nn.Conv3d(in_channels=complex, out_channels=1, kernel_size=1, stride=1, padding=0)
        else:
            self.last_conv = nn.Conv2d(in_channels=complex, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.last_sig  = nn.Sigmoid()

    def forward(self, x):
        encoder_features, skip_connections = self.encoder_series(x)
        bottle_features = self.bottleneck(encoder_features)
        decoder_features = self.decoder_series(bottle_features, skip_connections)
        final_conv_features = self.last_conv(decoder_features)
        output_features = self.last_sig(final_conv_features)
        return output_features

class UNet3D(GeneralUNet):
    def __init__(self, in_channels, conv_kernel_size, pool_kernel_size, up_kernel_size, dropout, conv_stride, conv_padding, size, complex):
        super(UNet3D, self).__init__(in_channels=in_channels, 
                                     conv_kernel_size=conv_kernel_size, 
                                     pool_kernel_size=pool_kernel_size, 
                                     up_kernel_size=up_kernel_size, 
                                     dropout=dropout, 
                                     conv_stride=conv_stride, 
                                     conv_padding=conv_padding, 
                                     conv3d=True, 
                                     size=size, 
                                     complex=complex)