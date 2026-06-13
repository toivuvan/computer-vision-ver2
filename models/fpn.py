import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=5):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.cv2 = ConvBNAct(hidden * 4, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class C2PSALite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.reduce = ConvBNAct(channels, hidden, kernel_size=1)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.Sigmoid(),
        )
        self.expand = ConvBNAct(hidden, channels, kernel_size=1)

    def forward(self, x):
        y = self.reduce(x)
        y = y * self.spatial_attn(y)
        return x + self.expand(y)


class FPN(nn.Module):
    """YOLO-like PAN-FPN neck with SPPF and lightweight spatial attention.

    The output contract stays compatible with the existing FCOS head:
    {"p3": stride-8, "p4": stride-16, "p5": stride-32}.
    """

    def __init__(self, in_channels=[192, 384, 768], out_channels=256):
        super().__init__()
        self.lateral_c3 = ConvBNAct(in_channels[0], out_channels, kernel_size=1)
        self.lateral_c4 = ConvBNAct(in_channels[1], out_channels, kernel_size=1)
        self.lateral_c5 = ConvBNAct(in_channels[2], out_channels, kernel_size=1)

        self.sppf = SPPF(out_channels, out_channels)
        self.c2psa = C2PSALite(out_channels)

        self.top_p4 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)
        self.top_p3 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)

        self.down_p3 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
        self.pan_p4 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)
        self.down_p4 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
        self.pan_p5 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)

    def forward(self, features):
        c3 = features["c3"]
        c4 = features["c4"]
        c5 = features["c5"]

        c3_lat = self.lateral_c3(c3)
        c4_lat = self.lateral_c4(c4)
        c5_lat = self.lateral_c5(c5)

        p5_td = self.c2psa(self.sppf(c5_lat))
        p4_td = self.top_p4(torch.cat([c4_lat, F.interpolate(p5_td, size=c4_lat.shape[2:], mode="nearest")], dim=1))
        p3 = self.top_p3(torch.cat([c3_lat, F.interpolate(p4_td, size=c3_lat.shape[2:], mode="nearest")], dim=1))

        p4 = self.pan_p4(torch.cat([p4_td, self.down_p3(p3)], dim=1))
        p5 = self.pan_p5(torch.cat([p5_td, self.down_p4(p4)], dim=1))

        return {"p3": p3, "p4": p4, "p5": p5}
