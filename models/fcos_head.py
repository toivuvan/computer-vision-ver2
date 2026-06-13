import math
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.cv1 = ConvBNAct(channels, channels, kernel_size=3)
        self.cv2 = ConvBNAct(channels, channels, kernel_size=3)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C3k2Lite(nn.Module):
    """Small CSP/C3k2-style block inspired by YOLO11 modules."""

    def __init__(self, channels, num_blocks=1):
        super().__init__()
        hidden = channels // 2
        self.cv1 = ConvBNAct(channels, hidden, kernel_size=1)
        self.cv2 = ConvBNAct(channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[Bottleneck(hidden) for _ in range(num_blocks)])
        self.cv3 = ConvBNAct(hidden * 2, channels, kernel_size=1)

    def forward(self, x):
        return self.cv3(torch.cat([self.blocks(self.cv1(x)), self.cv2(x)], dim=1))


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    """YOLO11-like decoupled anchor-free head.

    It keeps the repository output contract: class logits and l/t/r/b distances.
    Unlike the old FCOS head, it does not predict centerness/objectness.
    """

    def __init__(self, in_channels=256, num_classes=5, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.cls_towers = nn.ModuleList()
        self.reg_towers = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.scales = nn.ModuleList()

        for _ in strides:
            self.cls_towers.append(nn.Sequential(
                C3k2Lite(in_channels, num_blocks=1),
                ConvBNAct(in_channels, in_channels, kernel_size=3),
            ))
            self.reg_towers.append(nn.Sequential(
                C3k2Lite(in_channels, num_blocks=1),
                ConvBNAct(in_channels, in_channels, kernel_size=3),
            ))
            self.cls_preds.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))
            self.reg_preds.append(nn.Conv2d(in_channels, 4, kernel_size=1))
            self.scales.append(Scale(1.0))

        prior_prob = 0.01
        bias_val = -math.log((1 - prior_prob) / prior_prob)
        for cls_pred, reg_pred in zip(self.cls_preds, self.reg_preds):
            nn.init.constant_(cls_pred.bias, bias_val)
            nn.init.normal_(cls_pred.weight, std=0.01)
            nn.init.constant_(reg_pred.bias, 0.0)
            nn.init.normal_(reg_pred.weight, std=0.01)

    def forward(self, features):
        cls_scores = []
        bbox_preds = []
        centerness_preds = []

        for idx, x in enumerate(features):
            cls_feat = self.cls_towers[idx](x)
            reg_feat = self.reg_towers[idx](x)

            cls_scores.append(self.cls_preds[idx](cls_feat))

            reg_output = self.scales[idx](self.reg_preds[idx](reg_feat))
            bbox_pred = torch.exp(reg_output).clamp(max=128.0) * float(self.strides[idx])
            bbox_preds.append(bbox_pred)

            # Kept only for backward-compatible flattening code. Not used in loss/inference.
            centerness_preds.append(torch.zeros(
                x.shape[0], 1, x.shape[2], x.shape[3],
                dtype=x.dtype,
                device=x.device,
            ))

        return cls_scores, bbox_preds, centerness_preds
