import torch
import torch.nn as nn
from layers import EncoderBlock, DecoderBlcok, EncoderUNet, DecoderUNet

class GeneralUNet(nn.Module):
    '''
    General class for UNet Structure
    '''
    def __init__(self, in_channels, out_channels, size, args):
        super(GeneralUNet, self).__init__()


    def forward(self, x):
        encoder_features = self.encoder(x)
        output_features = self.decoder(encoder_features)
        return output_features

class UNet3D(GeneralUNet):
    def __init__(self, moreargs):
        super(UNet3D, self).__init__()