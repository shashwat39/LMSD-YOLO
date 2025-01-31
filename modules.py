import torch
import torch.nn as nn
import torch.nn.functional as F


class AconC(nn.Module):
    r""" ACON activation (activate or not)
    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # auto-pad
    return p

class DSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, groups=1, act=True, bias=False):
        super(DSConv, self).__init__()
        self.depthwiseconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=bias)  # Depthwise conv
        self.pointwiseconv = nn.Conv2d(c1, c2, kernel_size=1)  # Pointwise conv
        self.bn = nn.BatchNorm2d(c2)
        self.act = AconC(c1=c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.pointwiseconv(self.depthwiseconv(x))))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemBlock, self).__init__()
        self.dsc1 = DSConv(in_channels, out_channels, s=2)  # Pass stride as 's'

        self.branch1_first_block = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.branch1_second_block = DSConv(out_channels, out_channels)  # Removed stride argument
        self.branch1_second_block_bn = nn.BatchNorm2d(out_channels)
        self.branch1_second_block_aconc = AconC(out_channels)
        self.branch2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x1 = self.dsc1(x)
        branch1_1 = self.branch1_first_block(x1)
        branch1_2 = self.branch1_second_block(branch1_1)
        branch1_2 = self.branch1_second_block_bn(branch1_2)
        branch1_2 = self.branch1_second_block_aconc(branch1_2)
        branch2 = self.branch2_maxpool(x1)
        
        # Resize if necessary (e.g., if spatial dimensions don't match)
        if branch1_2.shape[2:] != branch2.shape[2:]:
            branch2 = F.interpolate(branch2, size=branch1_2.shape[2:], mode='bilinear', align_corners=False)
        
        concatenated = torch.cat([branch1_2, branch2], dim=1)
        out = self.concat_conv(concatenated)
        return out

# Example usage
in_channels = 3
out_channels = 64
stem_block = StemBlock(in_channels, out_channels)
input_tensor = torch.randn(1, in_channels, 224, 224)
output_tensor = stem_block(input_tensor)
print(output_tensor.shape)  # Should be (1, out_channels, 56, 56)

class DBAModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DBAModule, self).__init__()
        self.dsc = DSConv(in_channels, out_channels, k=3, s=stride, groups=in_channels)  # Ensure groups is set correctly
        self.bn = nn.BatchNorm2d(out_channels)
        self.aconc = AconC(out_channels)

    def forward(self, x):
        x = self.dsc(x)
        x = self.bn(x)
        x = self.aconc(x)
        return x


# Create a DBA module with 64 input channels and 128 output channels
dba_module = DBAModule(in_channels=64, out_channels=128)

# Create a random input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(16, 64, 32, 32)

# Forward pass through the DBA module
output_tensor = dba_module(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class GAM_Attention(nn.Module):
   #https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True,rate=4):
        super(GAM_Attention, self).__init__()
        
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )
        
        
        self.spatial_attention = nn.Sequential(
            
            nn.Conv2d(c1, c1//rate, kernel_size=7, padding=3,groups=rate)if group else nn.Conv2d(c1, int(c1 / rate), kernel_size=7, padding=3), 
            nn.BatchNorm2d(int(c1 /rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1//rate, c2, kernel_size=7, padding=3,groups=rate) if group else nn.Conv2d(int(c1 / rate), c2, kernel_size=7, padding=3), 
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
       # x_channel_att=channel_shuffle(x_channel_att,4) #last shuffle 
        x = x * x_channel_att
 
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att=channel_shuffle(x_spatial_att,4) #last shuffle 
        out = x * x_spatial_att
        #out=channel_shuffle(out,4) #last shuffle 
        return out  




class DWASFFV5(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        ASFF version for YoloV5.
        different than YoloV3
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be 
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you need change code manually.
        """
        super(DWASFFV5, self).__init__()
        self.level = level
        self.dim = [int(1024*multiplier), int(512*multiplier),
                    int(256*multiplier)]
        # print(self.dim)
        
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = DSConv(int(512*multiplier), self.inter_dim, 3, 2)
                
            self.stride_level_2 = DSConv(int(256*multiplier), self.inter_dim, 3, 2)
                
            self.expand = DSConv(self.inter_dim, int(
                1024*multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = DSConv(
                int(1024*multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = DSConv(
                int(256*multiplier), self.inter_dim, 3, 2)
            self.expand = DSConv(self.inter_dim, int(512*multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = DSConv(
                int(1024*multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = DSConv(
                int(512*multiplier), self.inter_dim, 1, 1)
            self.expand = DSConv(self.inter_dim, int(
                256*multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory

        ###计算权重过程
        compress_c = 8 if rfb else 16
        self.weight_level_0 = DSConv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = DSConv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = DSConv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = DSConv(
            compress_c*3, 3, 1, 1)
        self.vis = vis
        self.gam = GAM_Attention(compress_c,compress_c)

    def forward(self, x): #l,m,s
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0=x[2] #l
        x_level_1=x[1] #m
        x_level_2=x[0] #s
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
            # level_0_resized = self.gam(level_0_resized)  #使用gam注意力机制

            level_1_resized = self.stride_level_1(x_level_1)
            # level_1_resized = self.gam(level_1_resized)

            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
            # level_2_resized = self.gam(level_2_resized)


            ####添加注意力机制gam
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
            ##
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
        #      level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)


        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] +\
            level_1_resized * levels_weight[:, 1:2, :, :] +\
            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


# Create an instance of DWASFFV5
level = 1  # Choose the level (0, 1, or 2)
multiplier = 1  # Choose the multiplier (1 or 0.5)
model = DWASFFV5(level=level, multiplier=multiplier)

# Prepare input tensors
# Example shapes: (batch_size, channels, height, width)
batch_size = 1
channels_l = int(1024 * multiplier)  # Level 0
channels_m = int(512 * multiplier)   # Level 1
channels_s = int(256 * multiplier)   # Level 2
# Prepare input tensors with aligned spatial dimensions
height, width = 64, 64  # Ensure dimensions are divisible by 4
x_level_0 = torch.randn(batch_size, channels_l, height // 2, width // 2)  # Large feature map
x_level_1 = torch.randn(batch_size, channels_m, height, width)           # Medium feature map
x_level_2 = torch.randn(batch_size, channels_s, height * 2, width * 2)   # Small feature map

# Combine inputs into a list
x = [x_level_2, x_level_1, x_level_0]  # Order: small, medium, large

# Pass inputs through the model
output = model(x)

# Check the output
print("Output shape:", output.shape)

# Ref - https://em-1001.github.io/computer%20vision/SCYLLA-IoU/ 

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint", iou_mode="SIoU", eps=1e-7):
    """
    Compute Intersection over Union (IoU) or SCYLLA-IoU (SIoU) between predicted and ground truth boxes.

    Args:
        boxes_preds (torch.Tensor): Predicted bounding boxes.
        boxes_labels (torch.Tensor): Ground truth bounding boxes.
        box_format (str): Format of the boxes ("midpoint" or "corners").
        iou_mode (str): IoU mode ("SIoU" or standard IoU).
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: IoU or SIoU values.
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    else:
        raise ValueError("Invalid box_format. Use 'midpoint' or 'corners'.")

    # Width and height of boxes
    w1, h1 = box1_x2 - box1_x1, box1_y2 - box1_y1
    w2, h2 = box2_x2 - box2_x1, box2_y2 - box2_y1

    # Intersection coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Intersection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Union area
    box1_area = abs(w1 * h1)
    box2_area = abs(w2 * h2)
    union = box1_area + box2_area - intersection + eps

    # IoU
    iou = intersection / union

    if iou_mode == "SIoU":
        # Convex hull dimensions
        C_w = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
        C_h = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)

        # Center distance
        C_x = torch.abs(box2_x1 + box2_x2 - box1_x1 - box1_x2) * 0.5
        C_y = torch.abs(box2_y1 + box2_y2 - box1_y1 - box1_y2) * 0.5
        p2 = (C_x ** 2 + C_y ** 2)

        # SIoU penalty terms
        sigma = torch.sqrt(p2) + eps
        sin_alpha = torch.clamp(C_y / sigma, -1, 1)
        sin_beta = torch.clamp(C_x / sigma, -1, 1)
        sin_alpha = torch.where(sin_alpha > torch.sqrt(torch.tensor(0.5)), sin_beta, sin_alpha)
        Lambda = torch.sin(2 * torch.arcsin(sin_alpha))
        gamma = 2 - Lambda
        rho_x = (C_x / (C_w + eps)) ** 2
        rho_y = (C_y / (C_h + eps)) ** 2
        Delta = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)
        omega_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omega_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        Omega = torch.pow(1 - torch.exp(-omega_w), 4) + torch.pow(1 - torch.exp(-omega_h), 4)
        R_siou = (Delta + Omega) * 0.5

        return iou - R_siou
    else:
        return iou

class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemBlock, self).__init__()
        self.dsc1 = DSConv(in_channels, out_channels, s=2)
        self.branch1_first_block = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.branch1_second_block = DSConv(out_channels, out_channels)
        self.branch1_second_block_bn = nn.BatchNorm2d(out_channels)
        self.branch1_second_block_aconc = AconC(out_channels)
        self.branch2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x1 = self.dsc1(x)
        branch1_1 = self.branch1_first_block(x1)
        branch1_2 = self.branch1_second_block(branch1_1)
        branch1_2 = self.branch1_second_block_bn(branch1_2)
        branch1_2 = self.branch1_second_block_aconc(branch1_2)
        branch2 = self.branch2_maxpool(x1)
        
        if branch1_2.shape[2:] != branch2.shape[2:]:
            branch2 = F.interpolate(branch2, size=branch1_2.shape[2:], mode='bilinear', align_corners=False)
        
        concatenated = torch.cat([branch1_2, branch2], dim=1)
        out = self.concat_conv(concatenated)
        return out


class Hsigmoid(nn.Module):
    """
    Hard sigmoid function
    """
    def __init__(self, inplace: bool = True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu6(input + 3.0, inplace=self.inplace) * (1.0/6.0)


class Hardswish(nn.Module):
    # Hard-SiLU activation
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for TorchScript and CoreML
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for TorchScript, CoreML and ONNX


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MobileNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_channels, stride, se=False, nl='Relu'):
        super(MobileNeck, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = se
        if se:
            self.se_block = SELayer(out_channels)
        if nl == 'Relu':
            self.activation = nn.ReLU(inplace=True)
        elif nl == 'ACON-C':
            self.activation = AconC(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if self.se:
            x = self.se_block(x)
        x = self.activation(x)
        return x

class SMobileNet(nn.Module):
    def __init__(self):
        super(SMobileNet, self).__init__()
        self.stem = StemBlock(in_channels=3, out_channels=16)
        self.neck1 = MobileNeck(in_channels=16, out_channels=32, kernel_size=3, hidden_channels=16, stride=2, se=False, nl='Relu')
        self.neck2 = MobileNeck(in_channels=32, out_channels=128, kernel_size=3, hidden_channels=72, stride=2, se=False, nl='Relu')
        self.neck3 = MobileNeck(in_channels=128, out_channels=64, kernel_size=5, hidden_channels=96, stride=2, se=True, nl='ACON-C')
        self.neck4 = MobileNeck(in_channels=64, out_channels=128, kernel_size=5, hidden_channels=240, stride=1, se=True, nl='ACON-C')
        self.neck5 = MobileNeck(in_channels=128, out_channels=256, kernel_size=3, hidden_channels=120, stride=2, se=True, nl='ACON-C')
        self.neck6 = MobileNeck(in_channels=256, out_channels=512, kernel_size=3, hidden_channels=288, stride=2, se=True, nl='ACON-C')
        
    def forward(self, x):
        x = self.stem(x)
        x = self.neck1(x)
        x = self.neck2(x)
        x = self.neck3(x)
        x = self.neck4(x)
        x = self.neck5(x)
        x = self.neck6(x)
        return x

model = SMobileNet()
input_tensor = torch.randn(1, 3, 640, 640)  # Batch of 1 image with 3 channels and 640x640 resolution
output = model(input_tensor)
print(output.shape) 


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")

    def forward(self, x):
        return self.upsample(x)

class Neck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Neck, self).__init__()

        # Define DBA blocks (5 times each)
        self.dba1 = nn.Sequential(*[DBAModule(in_channels, out_channels) for _ in range(5)])
        self.dba2 = DBAModule(out_channels, out_channels)
        self.dba3 = DBAModule(out_channels, out_channels)

        # Upsampling layers
        self.upsample1 = UpSample(out_channels)
        self.upsample2 = UpSample(out_channels)

        # Final DBA stacks after concatenation
        self.dba_final1 = nn.Sequential(*[DBAModule(out_channels * 2, out_channels) for _ in range(5)])
        self.dba_final2 = nn.Sequential(*[DBAModule(out_channels * 2, out_channels) for _ in range(5)])

    def forward(self, x1, x2, x3):
        """
        x1: High-resolution feature map (top)
        x2: Mid-resolution feature map (middle)
        x3: Low-resolution feature map (bottom)
        """

        # Process each path
        x1 = self.dba1(x1)

        x2 = self.dba2(x2)
        x2_up = self.upsample1(x2)
        x2_concat = torch.cat([x1, x2_up], dim=1)  # Concatenate with x1
        x2_out = self.dba_final1(x2_concat)

        x3 = self.dba3(x3)
        x3_up = self.upsample2(x3)
        x3_concat = torch.cat([x2_out, x3_up], dim=1)  # Concatenate with x2_out
        x3_out = self.dba_final2(x3_concat)

        return x1, x2_out, x3_out  # Three output feature maps

batch_size = 1
in_channels = 128  # Ensure this matches DBAModule expectations
out_channels = 128
height, width = 64, 64  # Highest resolution feature map size

x1 = torch.randn(batch_size, in_channels, height, width)        # High-resolution feature map
x2 = torch.randn(batch_size, in_channels, height//2, width//2)  # Mid-resolution feature map
x3 = torch.randn(batch_size, in_channels, height//4, width//4)  # Low-resolution feature map

# Initialize the Neck model
neck = Neck(in_channels, out_channels)

# Forward pass
output_x1, output_x2, output_x3 = neck(x1, x2, x3)

# Print output shapes
print("Output Shapes:")
print("x1_out:", output_x1.shape) 
print("x2_out:", output_x2.shape) 
print("x3_out:", output_x3.shape) 
