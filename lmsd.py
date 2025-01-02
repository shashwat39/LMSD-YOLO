import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ACONC(nn.Module):
    def __init__(self, channels):
        super(ACONC, self).__init__()
        self.p1 = nn.Parameter(torch.ones(1, channels, 1, 1))  # Initialize p1 to 1.0
        self.p2 = nn.Parameter(torch.zeros(1, channels, 1, 1))  # Initialize p2 to 0.0
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, groups=channels)

    def forward(self, x):
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        beta = self.conv1(mean)
        beta = self.conv2(beta)
        beta = torch.sigmoid(beta)
        p1_minus_p2 = self.p1 - self.p2
        acon_c = p1_minus_p2 * x * torch.sigmoid(beta * p1_minus_p2 * x) + self.p2 * x
        return acon_c

class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemBlock, self).__init__()
        self.dsc1 = DepthwiseSeparableConv(in_channels, out_channels, stride=2)
        self.branch1_first_block = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.branch1_second_block = DepthwiseSeparableConv(out_channels, out_channels, stride=2)  # Fixed input channels
        self.branch1_second_block_bn = nn.BatchNorm2d(out_channels)
        self.branch1_second_block_aconc = ACONC(out_channels)
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
        self.dsc = DepthwiseSeparableConv(in_channels, out_channels, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.aconc = ACONC(out_channels)
    
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

class DSASFF(nn.Module):
    def __init__(self, channels, levels=[128, 256, 512]):
        super(DSASFF, self).__init__()
        self.levels = levels
        self.dsc = DepthwiseSeparableConv(channels, channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=0)
        
        # Learnable parameters
        self.lambda_alpha = nn.Parameter(torch.randn(1))
        self.lambda_beta = nn.Parameter(torch.randn(1))
        self.lambda_gamma = nn.Parameter(torch.randn(1))
    
    def forward(self, x0, x1, x2):
        # Upsample x0 to match x1
        x0_compressed = self.dsc(x0)
        x0_upsampled = F.interpolate(x0_compressed, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Process x1 with depthwise convolution
        x1_processed = self.dsc(x1)
        
        # Downsample x2 to match x1
        x2_downsampled = self.maxpool(x2)
        x2_compressed = self.dsc(x2_downsampled)
        
        # Compute weights
        alpha = torch.exp(self.lambda_alpha) / (torch.exp(self.lambda_alpha) + torch.exp(self.lambda_beta) + torch.exp(self.lambda_gamma))
        beta = torch.exp(self.lambda_beta) / (torch.exp(self.lambda_alpha) + torch.exp(self.lambda_beta) + torch.exp(self.lambda_gamma))
        gamma = torch.exp(self.lambda_gamma) / (torch.exp(self.lambda_alpha) + torch.exp(self.lambda_beta) + torch.exp(self.lambda_gamma))
        
        # Combine feature maps
        out = alpha * x0_upsampled + beta * x1_processed + gamma * x2_compressed
        return out

# Example usage
channels = 64  # Example number of channels
dsasff = DSASFF(channels)
x0 = torch.randn(1, channels, 128, 128)  # Example input for level 0
x1 = torch.randn(1, channels, 256, 256)  # Example input for level 1
x2 = torch.randn(1, channels, 512, 512)  # Example input for level 2
output = dsasff(x0, x1, x2)
print(output.shape)

# Ref - https://em-1001.github.io/computer%20vision/SCYLLA-IoU/ 
import torch

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

# Assuming DepthwiseSeparableConv and ACONC are defined elsewhere
# class DepthwiseSeparableConv(nn.Module):
#     ...
# class ACONC(nn.Module):
#     ...

class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemBlock, self).__init__()
        self.dsc1 = DepthwiseSeparableConv(in_channels, out_channels, stride=2)
        self.branch1_first_block = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.branch1_second_block = DepthwiseSeparableConv(out_channels, out_channels, stride=2)
        self.branch1_second_block_bn = nn.BatchNorm2d(out_channels)
        self.branch1_second_block_aconc = ACONC(out_channels)
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


class Hswish(nn.Module):
    """
    Hard swish function
    """
    def __init__(self, inplace: bool = True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu6(input+3.0, inplace=self.inplace) * (1.0/6.0) * input


class Squeeze_excite(nn.Module):
    def __init__(self, num_channels, r=4):
        """
        Squeeze-and-Excitation block
          Args:
            num_channels (int): number of channels in the input tensor
            r (int): num_channels are divided by r in the first conv block
        """
        super(Squeeze_excite, self).__init__()

        #instead of fully connected layers 1x1 convolutions are used, which has exactly the same effect as the input tensor is 1x1 after pooling
        #batch normalization is not used here as it's absent in the paper
        self.conv_0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels//r, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_channels//r, num_channels, kernel_size=1),
            Hsigmoid()
        )

    def forward(self, input):
        out = self.conv_0(input)
        out = self.conv_1(out)
        out = out * input
        return out

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
            self.se_block = Squeeze_excite(out_channels)
        if nl == 'Relu':
            self.activation = nn.ReLU(inplace=True)
        elif nl == 'ACON-C':
            self.activation = ACONC(out_channels)
        
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

# Example usage
model = SMobileNet()
input_tensor = torch.randn(1, 3, 640, 640)  # Batch of 1 image with 3 channels and 640x640 resolution
output = model(input_tensor)
print(output.shape)  # Expected output shape: (1, 512, ...)