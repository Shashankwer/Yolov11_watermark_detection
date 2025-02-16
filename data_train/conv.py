import math
import numpy as np
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d>1:
        k = d*(k-1)+1 if isinstance(k, int) else [d*(x-1)  + 1 for x in k] # actual kernel size
    if p is None:
        p = k// 2 if isinstance(k, int) else [x//2 for x in k]
    return p

class Conv(nn.Module):
    """Standard convolution with args (ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)"""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1,s=1, p=None, g=1,d=1, act=True):
        """Initialize Conv layer with given arguments including activation"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=k, stride=s,padding= autopad(k,s,d), groups=g, dilation=d, bias=True)
        self.bn = nn.BatchNorm2d(c2)
        
        if act:
            self.act = self.default_act
        else: 
            if act is isinstance(act, nn.Module):
                act = act 
            else:
                act =  nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing"""
    def __init__(self, c1, c2, k=3,s=1, p=None, g=1, d=1,act=True):
        """Initialize Conv layer with given arguments including activation"""
        super().__init__(c1,c2,k,s,p,g=g,d=d,act=act)
        self.cv2 = nn.Conv2d(c1,c2,1,s,autopad(1,p,d), groups=g, dilation=d,bias=True)
    
    def forward(self,x):
        """Apply convolution, batch normalization and activation to input tensor"""
        return self.act(self.bn(self.conv(x)+ self.cv2(x)))
    
    def forward_fuse(self,x):
        """Apply fused convolution, batch normalization and activation to the input tensor"""
        return self.act(self.bn(self.conv(x)))
    
    def fuse_convs(self):
        """Fused parallel convolutions"""
        w = torch.zeros_like(self.conv.weight.data)
        i = [w//2 for x in w.shape[2:]]
        w[:,:,i[0]:i[0]+1, i[1]:i[1]+1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse
    

class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel)
    """
    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize the conv layer with given argument including activation"""
        super().__init__()
        self.conv1 = Conv(c1, c2,1, act=False)
        self.conv2 = DWConv(c2, c2,k,act=act)
    
    def forward(self, x):
        return self.conv2(self.conv1(x))


class DWConv(conv):
    """Depth-wise convolution"""
    def __init__(self, c1,c2, k=1, s=1,d=1, act=True):
        """Initialize the depthwise convolution network"""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1,c2))
    
class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer"""
    default_act = nn.SiLU()

    def __init__(self,c1, c2, k=2, s=2,p=0,bn=True, act=True):
        """Initialize ConvTranpose 2d layer"""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1,c2,k,s,p,bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self,x):
        """Applies activation and convolution transpose operation to input"""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self,x):
        """Applies activation and convolution transpose operation to input"""
        return self.act(self.conv_transpose(x))

class Focus(nn.Module):
    """Focus wh information into c-space"""
    def __init__(self, c1, c2, k=1,s=1,p=None,g=1, act=True):
        """Initializes focus object with user defined channel, convolution, padding, group and activation values"""
        super().__init__()
        self.conv = Conv(c1*4, c2, k,s,p,g,act=act)
    
    def forward(self,x):
        """
        Applies convolution to concatenated tensor and returns the output
        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2)
        """
        return self.conv(torch.cat((x[...,::2,::2],x[...,::2,::2],x[...,::2,::2],x[...,1::2,1::2]),1))

class GhostConv(nn.Module):
    """Ghost Convolution"""
    def __init__(self, c1,c2,k=1,s=1,g=1,act=True):
        super().__init__()
        c_ = c2//2
        self.cv1 = Conv(c1,c_,k,s,None, g, act=act)
        self.cv2 = Conv(c_,c_, 5,1,None, c_, act=act)
    
    def forward(self,x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)),1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training
    and deploy status

    This module is used in RT-DETR
    Based on https://github.com/
    """
    def __init__(self, c1, c2,k=1, s=1,g=1, act=True, bn=False, deploy=False):
        """Intializes Light Convolution layer with inputs, outputs & optional activation function"""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act,nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k,s,p=p, g=g,act=False)
        self.conv2 = Conv(c1,c2,1,s,p=(p-k//2),g=g,act=False)
    
    def forward_fuse(self,x):
        """Forward process"""
        return self.act(self.conv(x))

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, bais1d = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid , bias3x3 + bias1x1 + bias1d
    
    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1,[1,1,1,1])
        
    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network"""
        if branch is None:
            return 0,0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma= branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3,3),dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i,i%input_dim,1,1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
                running_mean = branch.running_mean
                running_var = branch.running_var
                gamma = branch.weight
                beta = branch.bias
                eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma/std).reshape(-1,1,1,1)
        return kernel * t, beta - running_mean * gamma
    
    def fuse_convs(self):
        """Combines two convolutions layers into a single layer and removes the unused attrubutes from the class"""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels= self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding = self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self,"bn"):
            self.__delattr__("bn")
        if hasattr(self,"nm"):
            self.__delattr__("nm")
        if hasattr(self,"id_tensor"):
            self.__delattr__("id_tensor")

class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, channels:int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels,1,1,0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x* self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument"""
        super().__init__()
        assert kernel_size in {3,7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2,1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
    
    def forward(self,x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x*self.act(self.cv1(torch.cat([torch.mean(x,1,keepdim=True),torch.max(x,1,keepdim=True)[0]],1)))

class CBAM(nn.Module):
    """Convolution Block Attention Module"""
    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size"""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
    

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension"""
    def __init__(self,dimension=1):
        """Concatenates a list of tensors along a specified dimension"""
        super().__init__()
        self.d = dimension
    
    def forward(self,x):
        """Forward pass for the YOLOv8 mask Proto module"""
        return torch.cat(x, self.d)

class Index(nn.Module):
    """Returns a particular index of the input."""
    def __init__(self, index=0):
        """Returns a particular index of the input."""
        super().__init__()
        self.index = index
    
    def forward(self,x):
        """
        Forward pass.
        Expects a list of tensors as input
        """
        return x[self.index]
