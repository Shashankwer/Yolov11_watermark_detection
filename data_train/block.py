import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import (Conv)

def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm 2d"""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True
        ).requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare Filter
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_eps)))
    fusedconv.weight.copy_(torch.mm(w_bn,w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weifght.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn,b_conv.reshape(1,-1)).reshape(-1)+b_bn)
    return fusedconv

class DFL(nn.Module):
    """
    Integral module of Distributed focal loss
    """
    def __init__(self, c1=16):
        """Initialize a convolution layer with a given number of input channels"""
        super().__init__()
        self.conv = nn.Conv2d(c1,1,1,bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1,c1,1,1))
        self.c1 = c1
    
    def forward(self,x):
        b,_, a = x.shape
        return self.conv(x.view(b,4,self.c1,a).transpose(2,1).softmax(1)).view(b,4,2)


class Proto(nn.Module):
    def __init__(self,c1, c_=256,c2=32):
        """
        Initializes the YOLOv8 mask Proto m,odule with a specified  number of protos
        and masks

        Input arguments are ch_in, number of protos, number of masks
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_,c_, 2,2,0, bias=True)
        self.cv2 = Conv(c_,c_, k=3)
        self.cv3 = Conv(c_, c2)
    
    def forward(self,x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3,3), e=0.5):
        super().__init__()
        c_ = int(c2*e)
        self.cv1 = Conv(c1, c_, k[0],1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1==c2
    
    def forward(self, x):
        """Applies the YOLO FPN to input data"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv2(x))

class C2f(nn.Module):
    """Faster implementation of CSP Bottleneck with 2 convolutions"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1,e=0.5):
        super().__init__()
        self.c = int(c2*e)
        self.cv1 = Conv(c1,2*self.c, 1,1)
        self.cv2 = Conv((2+n)*self.c, c2,1)
        self.m = nn.ModuleList(Bottleneck(self.c,self.c,shortcut,g,k=((3,3),(3,3)),e=1.0) for _ in range(n))

    def forward(self,x):
        """Forwsrding pass through C2f layer"""
        y = list(self.cv1(x).chunk(2,1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y,1))

    def forward_split(self,x):
        y = self.cv1(x).split((self.c, self.c),1)
        y = [y[0],y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y,1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolution"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP bottleneck with given channels, numbers, shortcut, groups and expansion value"""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1,c_, 1,1)
        self.cv2 = Conv(c1,c_, 1,1)
        self.cv3 = Conv(2*c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_,c_, shortcut, g,k=((1,1),(3,3)), e=1.0) for _ in range(n)))
    
    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions"""
        return self.cv3(torch.cat((self.m(self.cv1(x)),self.cv2(x)),1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural network"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configuration."""
        super().__init__(c1, c2, n, shortcut,g,e)
        c_ = int(c2*e)
        self.m = nn.Sequential(*(Bottleneck(c_,c_, shortcut,g,k=(k,k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """Faster implementation of CSP Bottleneck with 2 convolution"""
    def __init__(self,c1,c2,n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP bottleneck with 2 convolutions and optional C3k blocks"""
        super().__init__(c1,c2, n, shortcut, g,e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c,2,shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut,g) for _ in range(n)
        )

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5"""
    def __init__(self,c1,c2,k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size
        """
        super().__init__()
        c_ = c1//2 # hidden channel
        self.cv1 = Conv(c1, c_,1,1)
        self.cv2 = Conv(c_*4, c2, 1,1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
    
    def forward(self,x):
        """Forward pass through Ghost Convolution Block"""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y,1))

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor
    Args:
        dim(int): The input tensor dimension
        num_heads (int): The number of attention heads
        attn_ratio (float): The ratio of attention by any dimension
    
    Attributes:
        num_heads (int): The number of attention heads
        head_dim (int): The dimension of each attention head
        key_dim (int): The dimension of the attention key
        scale (float): The scaling factor for the attention scores
        qkv (Conv): Convolution layer for the computing key, query and value
        proj (Conv): Convolution layer for projecting the attended values
        pe (Conv): Convolution layer for positional encoding.
    """

    def __init__(self, dim, num_heads = 8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key and value convolution and positional encoding"""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.key_dim*attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h,1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3,1, g=dim,act=False)
    
    def forward(self, x):
        """
        Forward Pass for the attention Module

        Args: 
            x (torch.tensor): The input tensor
        
        Returns:
            (torch.Tensor): The output tensor after self-attention
        """
        B,C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q,k,v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2,-1)).view(B,C, H, W) + self.pe(v.reshape(B,C,H, W))
        x = self.proj(x)
        return x

class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention
    block for neural networks

    This class encapsulating the functionality for applying multi
    head attention and feedforward neural network layers with optional
    shortcut connections. 

    Attributes:
        attn (Attention): Multi head attention module
        ffn (nn.Sequential): Feed-forward neural network
        add (bool): Flag indicating whether to add shortcut conenction
    
    Methods:
        forward: Performs a forward pass through the PSA applying attention
            and feed-forward layers
    
    Examples:
        Creates a PSABlock and performs a forward pass
        >>> psaBlock = PSABlock(c=128,attn_ratio=0.5, num_heads=4, shortcut=False)
        >>> input_tensor = torch.randn(1,128,32,32)
        >>> output_tensor = psablock(input_tensor)
    """
    def __init__(self,c,attn_ratio=0.5, num_heads=4, shortcut=False) -> None:
        """Initializes the PSABlock with attention and feedforward layers for enhanced feature extraction"""
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c,c*2,1),Conv(c*2,c,1,act=False))
        self.add = shortcut
    
    def forward(self,x):
        """Execute a forward pass through PSABlock applying attention and feedforward layers to the input layer"""
        x = x + self.attn(x)  if self.add else self.attn(x)
        x = x + self.attn(x) if self.add else self.attn(x)
        return x


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing
    This module implements a convolution block with attention mechanism to enhance feature
    extraction and processing capabilities. It includes a series of PSABlock modules for self-attention

    Attributes:
        c(int): Number of hidden layers
        cv1(Conv): 1x1 convolution layer to reduce the number of input channels to 2*c
        cv2(Conv): 1x1 convolution layer to reduce the number of output channels in c
        m (nn.Sequential): Sequential container of PSABlock modules and feedforward operations
    
    Methods:
        forward: Performs a forward pass through the CSPSA module, applying attention and feedforward
    
    Notes:
        This module essentially is the same as a PSA module, but refactored to allow stacking PSA module
    
    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256,n=3, e=0.5)
        >>> input_tensor = torch.randn(1,256, 64,64)
        >>> output_tensor = c2psa(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio"""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2*self.c, 1,1)
        self.cv2 = Conv(2*self.c,c1,1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5,num_heads=self.c//64) for _ in range(n)))
    
    def forward(self,x):
        """Process the input tensor 'x' through a series of PSA blocks and return a transformed tensor"""
        a,b = self.cv1(x).split((self.c,self.c),dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a,b),1))
    
