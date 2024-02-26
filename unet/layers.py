import torch
import torch.nn as nn
import torch.nn.functional as F

def addBlock(sequence, in_channels, out_channels, kernel_size, dropout=0.0, stride=1, padding=0, conv3d=True):
    '''
    Adds a block to a neural network sequence based on a given configuration.
    
    Args:
        sequence (str): A string sequence consisting of 'c' (convolution), 'b' (batch normalization), 
                        'r' (ReLU activation), and 'd' (dropout) to specify the layers and their order.
        in_channels (int): Number of input channels for the convolutional layer.
        out_channels (int): Number of output channels for the convolutional layer.
        kernel_size (int or tuple): Size of the kernel for the convolutional layers.
        dropout (float, optional): Dropout rate for dropout layers. Default is 0.1.
        stride (int or tuple, optional): Stride for the convolutional layers. Default is 1.
        padding (int or tuple, optional): Padding for the convolutional layers. Default is 0.
        conv3d (bool, optional): Flag to determine whether to use 3D convolutions (True) or 2D convolutions (False). Default is True.

    Returns:
        modules (list of nn.Module): List of PyTorch modules representing the constructed convolutional block.
    '''
    modules = []
    for i in sequence:
        # Add Convolution layer (3D or 2D)
        if i == "c":
            if conv3d:
                print(f'adding conv3d layer with in_channels: {in_channels}, out_channels: {out_channels}')
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
                modules.append(nn.Dropout3d(p=dropout))
            else:
                modules.append(nn.Dropout2d(p=dropout))
    return modules

class ConvSingle(nn.Sequential):
    '''
    Single Convolutional Layer, encapsulating a standard set of layers: Convolution, Batch Normalization,
    ReLU activation, and optionally Dropout, based on the given architecture sequence.
    
    Args:
        in_channels (int): Number of input channels to the convolutional layer.
        out_channels (int): Number of output channels from the convolutional layer.
        kernel_size (int or tuple): Size of the convolutional kernel.
        dropout (float): Dropout rate. If dropout is 0, no dropout layer is added.
        stride (int or tuple): Stride for the convolutional layer.
        padding (int or tuple): Padding for the convolutional layer.
        conv3d (bool): Indicates whether to use 3D convolution (True) or 2D convolution (False).
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dropout, stride, padding, conv3d):
        super(ConvSingle, self).__init__()
        if dropout:
            for module in addBlock('cbrd', in_channels, out_channels, kernel_size, dropout, stride, padding, conv3d):
                self.add_module('conv_with_dropout', module)
        else:
            for module in addBlock('cbr', in_channels, out_channels, kernel_size, dropout, stride, padding, conv3d):
                self.add_module('conv_no_dropout', module)

class ConvDouble(nn.Sequential):
    '''
    Double Convolutional Layer module, comprising two sets of convolutional layers each followed by batch normalization
    and ReLU activation. Optionally, a dropout layer can be included after each ReLU activation.
    
    conv1 --> channel inc. or dec. based on encoder or not
    conv2 --> channel steady

    Args:
        in_channels (int): Number of input channels to the first convolutional layer.
        out_channels (int): Number of output channels from the second convolutional layer.
        kernel_size (int or tuple): Size of the convolutional kernels.
        endec (bool): Indicates whether the block is part of the encoder (True) or decoder (False). 
                      This influences intermediate channel sizes.
        dropout (float): Dropout rate for dropout layers.
        stride (int or tuple): Stride for the convolutional layers.
        padding (int or tuple): Padding for the convolutional layers.
        conv3d (bool): Indicates whether to use 3D convolutions (True) or 2D convolutions (False).
        first_enc (bool): True --> first encoder block, channel increase to 64, False --> Not first encoder block
    '''
    def __init__(self, in_channels, kernel_size, endec, dropout, stride, padding, conv3d, first=False):
        super(ConvDouble, self).__init__()

        if endec:
            if first:
                out_channels = 64
            else:
                out_channels = in_channels * 2
        else:                
            out_channels = int(in_channels // 2)
        print(f'Adding Double Conv layer, in: {in_channels}, out: {out_channels}')
        self.add_module("conv_single1", ConvSingle(in_channels, out_channels, kernel_size, dropout, stride, padding, conv3d))
        self.add_module("conv_single2", ConvSingle(out_channels, out_channels, kernel_size, dropout, stride, padding, conv3d))

class EncoderBlock(nn.Module):
    '''
    Encoder Block consisting of a double convolutional layer followed by a Max Pool operation. 
    This block also saves its output before pooling for use as a skip connection in a corresponding Decoder Block.
    
    Args:
        in_channels (int): Number of input channels to the block.
        out_channels (int): Number of output channels from the block.
        conv_kernel_size (int or tuple): Kernel size for the convolutional layers.
        pool_kernel_size (int or tuple): Kernel size for the max pooling operation.
        dropout (float): Dropout rate for dropout layers within the block.
        conv_stride (int or tuple): Stride for the convolutional layers.
        conv_padding (int or tuple): Padding for the convolutional layers.
        conv3d (bool): Indicates whether to use 3D convolutions and pooling (True) or 2D (False).
    '''
    def __init__(self, in_channels, conv_kernel_size, pool_kernel_size, dropout, conv_stride, conv_padding, conv3d, first):
        super(EncoderBlock, self).__init__()
        print(f'Encoder: adding convDouble')
        self.doubleConv = ConvDouble(in_channels, conv_kernel_size, True, dropout, conv_stride, conv_padding, conv3d, first=first)
        print(f'')
        if conv3d:
            self.maxPool = nn.MaxPool3d(pool_kernel_size)
        else:
            self.maxPool = nn.MaxPool2d(pool_kernel_size)
        # Skip Connection
        self.skip_features = None

    def forward(self, x):
        print(f'Encoder Block Forward')
        post_conv_features = self.doubleConv(x)
        print(f'post_conv_features.shape: {post_conv_features.shape}')
        # Skip Connection
        self.skip_features = post_conv_features
        post_pool_features = self.maxPool(post_conv_features)
        print(f'shape is: {post_pool_features.shape}')
        return post_pool_features

class DecoderBlock(nn.Module):
    '''
    Decoder Block consisting of an upscaling operation followed by a double convolutional layer. 
    This block combines upsampled features with features from a corresponding Encoder Block via a skip connection.
    
    Args:
        in_channels (int): Number of input channels to the block.
        out_channels (int): Number of output channels from the block.
        conv_kernel_size (int or tuple): Kernel size for the convolutional layers.
        up_kernel_size (int or tuple): Kernel size for the transposed convolutional layer used for upscaling.
        dropout (float): Dropout rate for dropout layers within the block.
        conv_stride (int or tuple): Stride for the convolutional layers.
        conv_padding (int or tuple): Padding for the convolutional layers.
        conv3d (bool): Indicates whether to use 3D transposed convolutions and convolutions (True) or 2D (False).
    '''
    def __init__(self, in_channels, conv_kernel_size, up_kernel_size, dropout, conv_stride, conv_padding, conv3d):
        super(DecoderBlock, self).__init__()
        print(f'Decoder: adding convDouble')
        out_channels = int(in_channels // 2)
        self.doubleConv = ConvDouble(in_channels, conv_kernel_size, False, dropout, conv_stride, conv_padding, conv3d)
        print(f'')
        if conv3d:
            self.upScale = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=up_kernel_size, stride=2)
        else:
            self.upScale = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=up_kernel_size, stride=2)

    def forward(self, x, skip_connection):
        print(f'Decoder Block Forward')
        post_ups_features = self.upScale(x)

        # padding decoder after upscaling
        diff = skip_connection.size(-1) - post_ups_features.size(-1)
        post_ups_features = F.pad(post_ups_features, (0, diff), "constant", 0)

        cat_features = torch.cat((post_ups_features, skip_connection), dim=1)

        post_conv_features = self.doubleConv(cat_features)

        return post_conv_features

class EncoderUNet(nn.Module):
    '''
    UNet Encoder composed of a series of Encoder Blocks. Each block applies a double convolution followed by max pooling,
    and stores the pre-pooled feature maps for use as skip connections in the decoder.
    
    Args:
        in_channels (int): Number of input channels to the first encoder block.
        out_channels (int): Number of output channels for each encoder block (assumes incrementing by block).
        conv_kernel_size (int or tuple): Convolutional kernel size for all encoder blocks.
        pool_kernel_size (int or tuple): Pooling kernel size for all encoder blocks.
        dropout (float): Dropout rate applied within each encoder block.
        conv_stride (int or tuple): Stride for the convolutional operations within each block.
        conv_padding (int or tuple): Padding for the convolutional operations within each block.
        conv3d (bool): Flag indicating whether the encoder uses 3D convolution (True) or 2D (False).
        size (int): Number of encoder blocks in the UNet.
    '''
    def __init__(self, in_channels, conv_kernel_size, pool_kernel_size, dropout, conv_stride, conv_padding, conv3d, size):
        super(EncoderUNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(EncoderBlock(in_channels, conv_kernel_size, pool_kernel_size, dropout, conv_stride, conv_padding, conv3d, first=True))
        in_channels = 64
        for i in range(size-1):
            self.blocks.append(EncoderBlock(in_channels, conv_kernel_size, pool_kernel_size, dropout, conv_stride, conv_padding, conv3d, first=False))
            in_channels *= 2

    def forward(self, x):
        skip_connections = []
        for block in self.blocks:
            x = block(x)
            skip_connections.append(block.skip_features)
        print(f'')
        print(f'Completed Encoder Block Forward Prop')
        print(f'x.shape: {x.shape}')
        for i in skip_connections:
            print(f'skip_connection.shape is: {i.shape}')
        return x, skip_connections

class DecoderUNet(nn.Module):
    '''
    UNet Decoder composed of a series of Decoder Blocks. Each block upscales the feature map and merges it with a corresponding
    feature map from the Encoder through skip connections, followed by a double convolution.
    
    Args:
        in_channels (int): Number of input channels to the first decoder block.
        out_channels (int): Number of output channels for each decoder block (assumes decrementing by block).
        conv_kernel_size (int or tuple): Convolutional kernel size for all decoder blocks.
        up_kernel_size (int or tuple): Upscaling kernel size for all decoder blocks.
        dropout (float): Dropout rate applied within each decoder block.
        conv_stride (int or tuple): Stride for the convolutional operations within each block.
        conv_padding (int or tuple): Padding for the convolutional operations within each block.
        conv3d (bool): Flag indicating whether the decoder uses 3D convolution (True) or 2D (False).
        size (int): Number of decoder blocks in the UNet.
    '''
    def __init__(self, in_channels, conv_kernel_size, up_kernel_size, dropout, conv_stride, conv_padding, conv3d, size):
        super(DecoderUNet, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(size):
            self.blocks.append(DecoderBlock(in_channels, conv_kernel_size, up_kernel_size, dropout, conv_stride, conv_padding, conv3d))
            in_channels = int(in_channels // 2)

    def forward(self, x, skip_connections):
        for block, skip_connection in zip(self.blocks, reversed(skip_connections)):
            x = block(x, skip_connection)
        return x
    
class Bottleneck(nn.Module):
    '''
    Bottleneck layer used in U-Net architecture for processing high-level features. This layer can operate in both 2D and 3D modes.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int or tuple): Size of the kernel in the convolutional layers.
        stride (int or tuple): Stride for the convolutional operations.
        padding (int or tuple): Padding for the convolutional operations.
        dropout (float, optional): Dropout rate; if specified, dropout is applied after batch normalization. Default: None (no dropout).
        conv3d (bool): Flag indicating whether to use 3D convolutions (True) or 2D convolutions (False).
    '''
    def __init__(self, in_channels, kernel_size, stride, padding, dropout=0, conv3d=True):
        super(Bottleneck, self).__init__()
        out_channels = in_channels * 2
        if conv3d:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
            self.relu2 = nn.ReLU(inplace=True)
            self.bn    = nn.BatchNorm3d(out_channels)
            self.dropout = nn.Dropout3d(p=dropout)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
            self.relu2 = nn.ReLU(inplace=True)
            self.bn    = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(p=dropout)

        print(f'Bottle neck appended, in: {in_channels}, out: {out_channels}')
        print(f'')

    def forward(self, x):
        print(f'Bottleneck Forward')
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x