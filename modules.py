import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


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

# BACKBONE
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
    
# NECK
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
        
        self.dba2 = DBAModule(in_channels, out_channels)
        self.dba3 = DBAModule(in_channels, out_channels)
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

        # Process x2 with upsampling and dba1
        x2 = self.dba2(x2)
        x2 = self.upsample1(x2)
        xn1 = self.neck1(xn1)
        x2 = torch.cat([x1, x2], dim=1)  # Concatenate with x1
        x2 = torch.cat([x2, xn1], dim=1)
        x2 = self.dba_final1(x2)

        # Process x3 with upsampling and dba2
        x3 = self.dba3(x3)
        x3 = self.upsample2(x3)
        xn2 = self.neck1(xn2)
        x3 = torch.cat([x2, x3], dim=1)  # Concatenate with x2
        x3 = torch.cat([x3, xn2], dim=1)
        x3_out = self.dba_final2(x3)

        return x1, x2, x3_out  # Three output feature maps

# PREDICTION

def non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.4):
    """
    Performs non-maximum suppression (NMS) on inference results

    Args:
        predictions (torch.Tensor): Tensor of shape (num_detections, 6) where each row contains
                                    [x1, y1, x2, y2, confidence, class_id].
        conf_threshold (float): Confidence threshold to filter detections.
        iou_threshold (float): IoU threshold for NMS.
        
    Returns:
        torch.Tensor: Detections after NMS. Each row has the format
                      [x1, y1, x2, y2, confidence, class_id].
    """
    # Ensure predictions are on the same device
    device = predictions.device

    # Filter out predictions with low confidence
    mask = predictions[:, 4] >= conf_threshold
    predictions = predictions[mask]
    
    if predictions.size(0) == 0:
        return torch.empty((0, 6), device=device)
    
    # This will hold the final detections after NMS for all classes
    final_detections = []

    # Get the unique class ids present in the predictions
    unique_labels = predictions[:, 5].unique()

    for cls in unique_labels:
        # Get detections for the current class
        cls_mask = predictions[:, 5] == cls
        cls_preds = predictions[cls_mask]

        # Boxes for NMS: shape (N, 4)
        boxes = cls_preds[:, :4]
        # Scores for NMS: shape (N,)
        scores = cls_preds[:, 4]

        # Perform NMS for the current class
        keep_indices = nms(boxes, scores, iou_threshold)
        final_detections.append(cls_preds[keep_indices])

    # Concatenate all detections from all classes
    if final_detections:
        final_detections = torch.cat(final_detections, dim=0)
    else:
        final_detections = torch.empty((0, 6), device=device)

    # Optionally, you can sort the final detections by confidence score (highest first)
    final_detections = final_detections[final_detections[:, 4].argsort(descending=True)]

    return final_detections


class PredictionModule(nn.Module):
    def __init__(self, out_channels_fuse, out_channels_dba, multiplier=1):
        """
        Args:
            out_channels_fuse (int): Number of channels output from the DWASFFV5 (fuse) branch.
                This is the value that DBAModule expects as input.
            out_channels_dba (int): Number of output channels from each DBAModule.
            multiplier (float): Channel multiplier for DWASFFV5 (if applicable).
        """
        super(PredictionModule, self).__init__()
        self.fusion1 = DWASFFV5(level=1, multiplier=multiplier)
        self.dba1    = DBAModule(in_channels=out_channels_fuse, out_channels=out_channels_dba)
        
        self.fusion2 = DWASFFV5(level=1, multiplier=multiplier)
        self.dba2    = DBAModule(in_channels=out_channels_fuse, out_channels=out_channels_dba)
        
        self.fusion3 = DWASFFV5(level=1, multiplier=multiplier)
        self.dba3    = DBAModule(in_channels=out_channels_fuse, out_channels=out_channels_dba)

    def forward(self, x):
        # Process the input through the first fusion and DBA
        x1 = self.fusion1(x)
        x1 = self.dba1(x1)

        # Process the input through the second fusion and DBA
        x2 = self.fusion2(x)
        x2 = self.dba2(x2)

        # Process the input through the third fusion and DBA
        x3 = self.fusion3(x)
        x3 = self.dba3(x3)

        # Combine the outputs from all three branches
        combined_output = torch.cat([x1, x2, x3], dim=1)

        # Apply NMS on the combined output
        final_predictions = non_max_suppression(combined_output)

        return final_predictions
