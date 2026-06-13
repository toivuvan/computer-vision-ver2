import torch
import torch.nn as nn
from .backbone import ConvNeXtTinyBackbone
from .fpn import FPN
from .fcos_head import FCOSHead

class FCOS(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = ConvNeXtTinyBackbone(pretrained=pretrained)
        self.fpn = FPN(in_channels=[192, 384, 768], out_channels=256)
        self.head = FCOSHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        # 1. Extract features from ConvNeXt backbone
        backbone_feats = self.backbone(x)
        
        # 2. Map features to P3, P4, P5 via FPN
        fpn_feats = self.fpn(backbone_feats)
        
        # 3. Pass through shared FCOS detection head
        # cls_scores, bbox_preds, centerness_preds are lists for P3, P4, P5
        cls_scores, bbox_preds, centerness_preds = self.head([fpn_feats["p3"], fpn_feats["p4"], fpn_feats["p5"]])

        # Flatten and concatenate predictions across all scales
        batch_size = x.shape[0]
        
        flat_cls_logits = []
        flat_bbox_preds = []
        flat_centerness_logits = []
        
        for cls_score, bbox_pred, centerness_pred in zip(cls_scores, bbox_preds, centerness_preds):
            # cls_score shape: B x C x H x W -> B x (H*W) x C
            B, C, H, W = cls_score.shape
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(B, H * W, C)
            flat_cls_logits.append(cls_score)
            
            # bbox_pred shape: B x 4 x H x W -> B x (H*W) x 4
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, H * W, 4)
            flat_bbox_preds.append(bbox_pred)
            
            # centerness_pred shape: B x 1 x H x W -> B x (H*W)
            centerness_pred = centerness_pred.permute(0, 2, 3, 1).reshape(B, H * W)
            flat_centerness_logits.append(centerness_pred)
            
        # Concatenate along the locations dimension
        cls_logits = torch.cat(flat_cls_logits, dim=1)        # B x 3549 x C
        bbox_preds = torch.cat(flat_bbox_preds, dim=1)        # B x 3549 x 4
        centerness_logits = torch.cat(flat_centerness_logits, dim=1) # B x 3549
        
        return {
            "cls_logits": cls_logits,
            "bbox_preds": bbox_preds,
            "centerness_logits": centerness_logits,
            "has_centerness": False,
        }
