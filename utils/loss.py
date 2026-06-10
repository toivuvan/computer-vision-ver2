import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCOSWithCIoULoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _focal_loss(self, logits, targets):
        # inputs: B * M x C logits
        # targets: B * M x C one-hot targets
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        return loss.sum()

    def _ciou_loss(self, pred_boxes, target_boxes):
        # pred_boxes, target_boxes: (N_pos, 4) in format [l, t, r, b]
        pl, pt, pr, pb = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        gl, gt, gr, gb = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

        # Predicted and ground truth areas
        pred_area = (pl + pr) * (pt + pb)
        target_area = (gl + gr) * (gt + gb)

        # Intersection width and height
        inter_w = torch.clamp(torch.min(pr, gr) + torch.min(pl, gl), min=0)
        inter_h = torch.clamp(torch.min(pb, gb) + torch.min(pt, gt), min=0)
        intersection = inter_w * inter_h

        # Union
        union = pred_area + target_area - intersection
        eps = 1e-6
        iou = intersection / (union + eps)

        # Enclosing box dimensions
        enc_w = torch.max(pr, gr) + torch.max(pl, gl)
        enc_h = torch.max(pb, gb) + torch.max(pt, gt)

        # Enclosing box diagonal squared
        c2 = enc_w ** 2 + enc_h ** 2 + eps

        # Center distance squared
        # center_x = x_c + (r - l) / 2
        # center_y = y_c + (b - t) / 2
        d2 = (
            ((pr - pl) - (gr - gl)) / 2.0
        ) ** 2 + (
            ((pb - pt) - (gb - gt)) / 2.0
        ) ** 2

        diou = iou - d2 / c2

        # CIoU aspect ratio term
        w_pred = pl + pr
        h_pred = pt + pb
        w_gt = gl + gr
        h_gt = gt + gb

        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(w_gt / (h_gt + eps)) - torch.atan(w_pred / (h_pred + eps)), 2
        )
        
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + eps)

        ciou = diou - alpha * v
        ciou_loss = 1.0 - ciou
        return ciou_loss

    def forward(self, predictions, cls_targets, reg_targets, centerness_targets):
        """
        predictions: dict with keys:
            "cls_logits": B x M x 5
            "bbox_preds": B x M x 4 (already exponentiated positive outputs)
            "centerness_logits": B x M
        cls_targets: B x M (values -1 to 4)
        reg_targets: B x M x 4
        centerness_targets: B x M
        """
        cls_logits = predictions["cls_logits"]
        bbox_preds = predictions["bbox_preds"]
        centerness_logits = predictions["centerness_logits"]

        batch_size, num_locs, num_classes = cls_logits.shape
        device = cls_logits.device

        # Reshape for focal loss
        flat_cls_logits = cls_logits.reshape(-1, num_classes)
        flat_cls_targets = cls_targets.reshape(-1)

        # Create one-hot class targets for Focal Loss (background gets all 0s)
        one_hot_targets = torch.zeros(flat_cls_logits.shape, device=device)
        pos_mask = flat_cls_targets >= 0
        if pos_mask.sum() > 0:
            one_hot_targets[pos_mask, flat_cls_targets[pos_mask]] = 1.0

        # 1. Classification Loss (Focal Loss over all pixels)
        loss_cls = self._focal_loss(flat_cls_logits, one_hot_targets)

        # 2. Regression & Centerness Losses (over positive pixels only)
        loss_reg = torch.tensor(0.0, device=device)
        loss_centerness = torch.tensor(0.0, device=device)
        
        # Flatten batch dimensions
        flat_bbox_preds = bbox_preds.reshape(-1, 4)
        flat_reg_targets = reg_targets.reshape(-1, 4)
        flat_centerness_logits = centerness_logits.reshape(-1)
        flat_centerness_targets = centerness_targets.reshape(-1)

        num_pos = pos_mask.sum().item()
        
        if num_pos > 0:
            pos_bbox_preds = flat_bbox_preds[pos_mask]
            pos_reg_targets = flat_reg_targets[pos_mask]
            
            # Box loss (CIoU)
            loss_reg = self._ciou_loss(pos_bbox_preds, pos_reg_targets).sum()

            # Centerness loss (BCE)
            pos_centerness_logits = flat_centerness_logits[pos_mask]
            pos_centerness_targets = flat_centerness_targets[pos_mask]
            loss_centerness = F.binary_cross_entropy_with_logits(
                pos_centerness_logits, pos_centerness_targets, reduction="sum"
            )

        # Normalize by positive locations count to keep loss scale stable
        norm_factor = max(num_pos, 1.0)
        
        total_loss = (loss_cls + loss_reg + loss_centerness) / norm_factor
        
        return {
            "loss": total_loss,
            "loss_cls": loss_cls / norm_factor,
            "loss_reg": loss_reg / norm_factor,
            "loss_centerness": loss_centerness / norm_factor,
            "num_pos": num_pos
        }
