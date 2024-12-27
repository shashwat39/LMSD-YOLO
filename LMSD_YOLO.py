import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        print(f"DepthwiseSeparableConv: Input shape = {x.shape}")
        x = self.depthwise(x)
        x = self.pointwise(x)
        print(f"DepthwiseSeparableConv: Output shape = {x.shape}")
        return x

class AconC(nn.Module):
    def __init__(self, width):
        super(AconC, self).__init__()
        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(f"ACON-C: Input shape = {x.shape}")
        output = (self.p1 * x - self.p2 * x) * self.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x
        print(f"ACON-C: Output shape = {output.shape}")
        return output

class DBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DBA, self).__init__()
        self.dsc = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.aconc = AconC(out_channels)

    def forward(self, x):
        print(f"DBA: Input shape = {x.shape}")
        x = self.dsc(x)
        x = self.bn(x)
        x = self.aconc(x)
        print(f"DBA: Output shape = {x.shape}")
        return x

class STEM(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(STEM, self).__init__()
        self.dsc1 = DepthwiseSeparableConv(input_channels, input_channels, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dsc2 = DepthwiseSeparableConv(input_channels, input_channels, kernel_size=3, stride=2, padding=1)
        self.conv1x1 = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1)
        self.bn_aconc = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            AconC(input_channels)
        )
        self.concat_conv = nn.Conv2d(input_channels * 2, output_channels, kernel_size=1, stride=1)

    def forward(self, x):
        print(f"STEM: Input shape = {x.shape}")
        path1 = self.dsc1(x)
        print(f"STEM: After DSC1, shape = {path1.shape}")
        path1_new = self.conv1x1(path1)
        print(f"STEM: After Conv1x1, shape = {path1_new.shape}")
        path1_new = self.dsc2(path1_new)
        print(f"STEM: After DSC2, shape = {path1_new.shape}")
        path1_new = self.bn_aconc(path1_new)
        print(f"STEM: After BatchNorm + ACON-C, shape = {path1_new.shape}")
        path2 = self.maxpool(path1)
        print(f"STEM: After MaxPool, shape = {path2.shape}")
        concatenated = torch.cat([path1_new, path2], dim=1)
        print(f"STEM: After concatenation, shape = {concatenated.shape}")
        output = self.concat_conv(concatenated)
        print(f"STEM: Final output shape = {output.shape}")
        return output

class DSA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSA, self).__init__()
        self.dba1 = DBA(in_channels, out_channels)
        self.dba2 = DBA(out_channels, out_channels)

    def forward(self, x):
        print(f"DSA: Input shape = {x.shape}")
        x = self.dba1(x)
        x = self.dba2(x)
        print(f"DSA: Output shape = {x.shape}")
        return x

class DWASFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWASFF, self).__init__()
        self.dba = DBA(in_channels, out_channels)

        self.upsample = nn.Identity()  

    def forward(self, x):
        print(f"DWASFF: Input shape = {x.shape}")
        x = self.dba(x)
        x = self.upsample(x)
        print(f"DWASFF: Output shape = {x.shape}")
        return x

class DBAUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DBAUpSample, self).__init__()
        self.dba = DBA(in_channels, out_channels)

        self.upsample = nn.Identity()  

    def forward(self, x):
        print(f"DBAUpSample: Input shape = {x.shape}")
        x = self.dba(x)
        x = self.upsample(x)
        print(f"DBAUpSample: Output shape = {x.shape}")
        return x

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.stem = STEM(input_channels=3, output_channels=16)
        self.dba_blocks = nn.Sequential(*[DBA(16, 16) for _ in range(5)])

    def forward(self, x):
        print("\nBackbone: Starting forward pass...")
        x = self.stem(x)
        print("\nBackbone: Stem output processed. Applying DBA blocks...")
        for i, block in enumerate(self.dba_blocks):
            print(f"Backbone: Applying DBA block {i+1}...")
            x = block(x)
        print(f"Backbone: Final output shape = {x.shape}")
        return x

class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.dsa1 = DSA(16, 32)
        self.dsa2 = DSA(32, 64)
        self.dwasff = DWASFF(64, 128)
        self.dba_upsample = DBAUpSample(128, 256)
        self.concat = nn.Conv2d(256 + 64, 512, kernel_size=1, stride=1, padding=0)
        self.dba_blocks = nn.Sequential(*[DBA(512, 512) for _ in range(5)])

    def forward(self, x):
        print("\nNeck: Starting forward pass...")
        print("Neck: Applying DSA1...")
        x1 = self.dsa1(x)
        print("Neck: Applying DSA2...")
        x2 = self.dsa2(x1)
        print("Neck: Applying DWASFF...")
        x3 = self.dwasff(x2)
        print("Neck: Applying DBAUpSample...")
        x4 = self.dba_upsample(x3)
        print("Neck: Resizing x2 to match x4 for concatenation...")
        x2_resized = F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=False)
        print(f"Neck: x2_resized shape = {x2_resized.shape}")
        print("Neck: Concatenating x4 and x2_resized...")
        x5 = torch.cat([x4, x2_resized], dim=1)
        print(f"Neck: Concatenated output shape = {x5.shape}")
        print("Neck: Applying 1x1 convolution to concatenated output...")
        x6 = self.concat(x5)
        print(f"Neck: Convolution output shape = {x6.shape}")
        print("Neck: Applying DBA blocks...")
        for i, block in enumerate(self.dba_blocks):
            print(f"Neck: Applying DBA block {i+1}...")
            x7 = block(x6)
        print(f"Neck: Final output shape = {x7.shape}")
        return x7

class Prediction(nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()
        self.conv = nn.Conv2d(512, 7, kernel_size=1, stride=1, padding=0)
        self.nms = nn.MaxPool2d(kernel_size=2, stride=2)  

    def forward(self, x):
        print("\nPrediction: Starting forward pass...")
        print("Prediction: Applying 1x1 convolution...")
        x = self.conv(x)
        print(f"Prediction: Convolution output shape = {x.shape}")
        print("Prediction: Applying NMS (MaxPool2d)...")
        x = self.nms(x)
        print(f"Prediction: Final output shape = {x.shape}")
        return x

class CompleteNetwork(nn.Module):
    def __init__(self):
        super(CompleteNetwork, self).__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.prediction = Prediction()

    def forward(self, x):
        print("\nCompleteNetwork: Starting forward pass...")
        print("CompleteNetwork: Processing through Backbone...")
        x = self.backbone(x)
        print("\nCompleteNetwork: Processing through Neck...")
        x = self.neck(x)
        print("\nCompleteNetwork: Processing through Prediction...")
        x = self.prediction(x)
        print("\nCompleteNetwork: Forward pass completed.")
        return x

if __name__ == "__main__":
    model = CompleteNetwork()

    x = torch.randn(1, 3, 640, 640)  
    print(f"\nInput shape: {x.shape}")

    output = model(x)
    print(f"\nFinal output shape: {output.shape}")