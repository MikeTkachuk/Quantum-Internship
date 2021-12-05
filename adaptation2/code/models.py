import torch
from torch import nn

import functools

class ConvBlock(nn.Module):
    def __init__(self, dim,
                 kernel_size=3,
                 stride=1,
                 activation=nn.Mish,
                 padding_type='reflect',
                 padding=1,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 use_bias=True,
                ):
        super(ConvBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, kernel_size, stride, activation, padding_type,padding, norm_layer, use_dropout, use_bias)
    
    def build_conv_block(self, dim, kernel_size, stride, activation, padding_type, padding, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (Mish))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(padding)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(padding)]
        elif padding_type == 'zero':
            p = padding
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.LazyConv2d(dim, kernel_size=kernel_size, stride=stride,padding=p, bias=use_bias), norm_layer(dim), activation()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)  
    
    def forward(self,x):
        return self.conv_block(x)
        

class DeconvBlock(nn.Module):
    def __init__(self, dim,
                 kernel_size=3,
                 stride=1,
                 activation=nn.Mish,
                 padding_type='reflect',
                 padding=1,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 use_bias=True,
                ):
        super(DeconvBlock, self).__init__()
        self.deconv_block = self.build_deconv_block(dim, kernel_size, stride, activation, padding_type,padding, norm_layer, use_dropout, use_bias)
    
    def build_deconv_block(self, dim, kernel_size, stride, activation, padding_type, padding, norm_layer, use_dropout, use_bias):
        """Construct a deconvolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (Mish))
        """
        deconv_block = []
        p = 0
        if padding_type == 'reflect':
            deconv_block += [nn.ReflectionPad2d(padding)]
        elif padding_type == 'replicate':
            deconv_block += [nn.ReplicationPad2d(padding)]
        elif padding_type == 'zero':
            p = padding
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        deconv_block += [nn.LazyConvTranspose2d(dim, kernel_size=kernel_size, stride=stride,padding=p, bias=use_bias), norm_layer(dim), activation()]
        if use_dropout:
            deconv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*deconv_block)  
    
    def forward(self,x):
        return self.deconv_block(x)
        
class ResBlock(nn.Module):
    """
    Define a Residual block with one Conv2D layer
    Credits: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/cf4191a3a4cc77fdffa5c0a8246c346049958e78/models/networks.py
    """

    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False):
        
        super(ResBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        
        
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (Mish))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.Mish(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)
    
    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
    
class DenseBlock(nn.Module):
    def __init__(self,
                 dim,
                 activation=nn.Mish,
                 norm_layer=nn.InstanceNorm1d,
                ):
        super(DenseBlock,self).__init__()
        
        if norm_layer == nn.BatchNorm2d:
            norm_layer = nn.BatchNorm1d
        if norm_layer == nn.InstanceNorm2d:
            norm_layer = nn.InstanceNorm1d
            
            
        if isinstance(norm_layer,nn.Identity):
            norm_layer = norm_layer()
        else:
            norm_layer = norm_layer(dim)
            
        modules = [
                   nn.LazyLinear(dim),
                   norm_layer,
                   activation(),
                  ] 
        self.module = nn.Sequential(*modules)
    
    def forward(self,x):
        return self.module(x)
    