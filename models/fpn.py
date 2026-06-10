import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels=[192, 384, 768], out_channels=256):
        super().__init__()
        # Lateral convolutions
        self.lateral_c3 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)
        
        # Smooth convolutions
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        c3 = features["c3"]
        c4 = features["c4"]
        c5 = features["c5"]

        # Top-down pathway
        p5_lat = self.lateral_c5(c5)  # B x 256 x 13 x 13
        p5 = self.smooth_p5(p5_lat)

        p4_lat = self.lateral_c4(c4)  # B x 256 x 26 x 26
        # Upsample P5 to match C4 size
        p5_up = F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p4_merged = p4_lat + p5_up
        p4 = self.smooth_p4(p4_merged)

        p3_lat = self.lateral_c3(c3)  # B x 256 x 52 x 52
        # Upsample P4 to match C3 size
        p4_up = F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p3_merged = p3_lat + p4_up
        p3 = self.smooth_p3(p3_merged)

        return {"p3": p3, "p4": p4, "p5": p5}
