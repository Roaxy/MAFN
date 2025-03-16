import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from arc import AdaptiveRotatedConv2d, RountingFunction
from functools import partial
from timm.models.helpers import named_apply
import torch.utils.checkpoint as checkpoint


class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2,args=None):
        super(SimpleDecoding, self).__init__()
        hidden_size = c4_dims//factor
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)
        print(args.kernels)
        self.mscb4 = MSRCB(hidden_size, hidden_size,stride=1,expansion_factor=1, dw_parallel=False, add=True,args=args)
        self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)


        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()

        self.mscb3 = MSRCB(hidden_size, hidden_size,stride=1, expansion_factor=1, dw_parallel=False, add=True,args=args)

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)

        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()


        self.mscb2 = MSRCB(hidden_size, hidden_size,stride=1, expansion_factor=1, dw_parallel=False, add=True,args=args)
        
        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)

        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()


        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, scale_factor=2, mode='bilinear', align_corners=True)
        # x_corr = self.Corr(x_c4, x_c3)
        # x = torch.cat([x_c4, x_corr, x_c3], dim=1)
        x = torch.cat([x_c4, x_c3], dim=1) ##torch.Size([2, 1536, 60, 60])
        # x = self.atten_4(x_c4,x_c3)

        x = self.conv1_4(x) ##torch.Size([2, 512, 60, 60])
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        
        x = self.mscb4(x)
        # x = self.conv2_4(x)
        # x = self.bn2_4(x)
        # x = self.relu2_4(x)

        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)##torch.Size([2, 512+256, 120, 120])
        # x = self.atten_3(x_c2,x)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.mscb3(x)
        # x = self.conv2_3(x)
        # x = self.bn2_3(x)
        # x = self.relu2_3(x)

        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)##torch.Size([2, 512+128, 240, 240])
        # x = self.atten_2(x_c1,x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.mscb2(x)
        # x = self.conv2_2(x)
        # x = self.bn2_2(x)
        # x = self.relu2_2(x)


        return self.conv1_1(x)
    
class AttentionGate(nn.Module):
    """Attention Gate
        https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                        stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                        stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class MSRCB(nn.Module):
    """
    Multi-scale Rotated convolution block (MSCB) 
    """
    def __init__(self, in_channels, out_channels, stride, expansion_factor=2, dw_parallel=True, add=True, activation='relu6',args=None):
        super(MSRCB, self).__init__()
        self.args=args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = self.args.kernels
        # self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            nn.ReLU(True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
            

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
        
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel


        routing_function2 = RountingFunction(in_channels=in_channels, kernel_number=1)
        self.conv2_3 =AdaptiveRotatedConv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=3, 
                                                               padding=1, rounting_func=routing_function2, bias=False, kernel_number=1)
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(True)
            )
            for kernel_size in self.kernel_sizes
        ])
        
    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = self.conv2_3(x)
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        return outputs
    
class Correlation(nn.Module):
    def __init__(self, max_disp=1, kernel_size=1, stride=1, use_checkpoint=False):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert stride == 1, "stride other than 1 is not implemented"
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.max_disp = max_disp
        self.padlayer = nn.ConstantPad2d(max_disp, 0)

    def forward_run(self, x_1, x_2):
        
        x_2 = self.padlayer(x_2)
        offsetx, offsety= torch.meshgrid([torch.arange(0, 2 * self.max_disp + 1),
                                                    torch.arange(0, 2 * self.max_disp + 1)], indexing='ij')
        
        w, h= x_1.shape[2], x_1.shape[3]
        x_out = torch.cat([torch.mean(x_1 * x_2[:, :, dx:dx+w, dy:dy+h], 1, keepdim=True)
                           for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))], 1)
        return x_out

    def forward(self, x_1, x_2):
        
        if self.use_checkpoint and x_1.requires_grad and x_2.requires_grad:
            x = checkpoint.checkpoint(self.forward_run, x_1, x_2)
        else:
            x = self.forward_run(x_1, x_2)
        return x