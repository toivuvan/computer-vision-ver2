import torch
import torch.nn as nn
import torchvision

class ConvNeXtTinyBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = torchvision.models.convnext_tiny(weights=weights)
        features = convnext.features
        
        # Split features into stages:
        # features[0:2]: Stem + Stage 1 -> Stride 4 (96 channels)
        # features[2:4]: Downsample + Stage 2 -> Stride 8 (192 channels) -> C3
        # features[4:6]: Downsample + Stage 3 -> Stride 16 (384 channels) -> C4
        # features[6:8]: Downsample + Stage 4 -> Stride 32 (768 channels) -> C5
        self.stage0_1 = features[0:2]
        self.stage2 = features[2:4]
        self.stage3 = features[4:6]
        self.stage4 = features[6:8]

    def forward(self, x):
        x = self.stage0_1(x)  # B x 96 x 104 x 104
        c3 = self.stage2(x)   # B x 192 x 52 x 52
        c4 = self.stage3(c3)  # B x 384 x 26 x 26
        c5 = self.stage4(c4)  # B x 768 x 13 x 13
        return {"c3": c3, "c4": c4, "c5": c5}
