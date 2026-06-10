import math
import torch
import torch.nn as nn

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, input):
        return input * self.scale

class FCOSHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=5, num_convs=4):
        super().__init__()
        self.num_classes = num_classes

        # Classification tower
        cls_tower = []
        for _ in range(num_convs):
            cls_tower.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)

        # Regression tower
        reg_tower = []
        for _ in range(num_convs):
            reg_tower.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)

        # Heads
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)

        # Scale layers for each feature map level (P3, P4, P5)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(3)])

        # Prior initialization for classification to stabilize warmup
        prior_prob = 0.01
        bias_val = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_val)
        nn.init.normal_(self.cls_pred.weight, std=0.01)

        # Initialize regression head
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)

        # Initialize centerness head
        nn.init.normal_(self.centerness.weight, std=0.01)
        nn.init.constant_(self.centerness.bias, 0.0)

    def forward(self, features):
        # features: List of tensors [p3, p4, p5]
        cls_scores = []
        bbox_preds = []
        centerness_preds = []

        for idx, x in enumerate(features):
            cls_feat = self.cls_tower(x)
            reg_feat = self.reg_tower(x)

            # Class scores & centerness
            cls_score = self.cls_pred(cls_feat)
            centerness = self.centerness(cls_feat)

            # Bounding box regression
            reg_output = self.reg_pred(reg_feat)
            # Apply scale layer and decode with exponential function
            reg_output = self.scales[idx](reg_output)
            # Decode to ensure regression values are positive
            bbox_pred = torch.exp(reg_output)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centerness_preds.append(centerness)

        return cls_scores, bbox_preds, centerness_preds
