import torch

def bbox_iou_tensor(boxes1, boxes2):
    # boxes1: N x 4, boxes2: M x 4 or 1 x 4 (format [xmin, ymin, xmax, ymax])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)

def soft_nms(boxes, scores, labels, sigma=0.5, score_threshold=0.001, confidence_threshold=0.05):
    """
    Per-class Soft-NMS implementation.
    boxes: N x 4 tensor [xmin, ymin, xmax, ymax]
    scores: N tensor
    labels: N tensor (class IDs)
    sigma: Gaussian decay parameter
    score_threshold: Lower bound to discard boxes
    confidence_threshold: Prediction output confidence limit
    """
    unique_labels = torch.unique(labels)
    
    keep_boxes = []
    keep_scores = []
    keep_labels = []
    
    for c in unique_labels:
        mask = labels == c
        c_boxes = boxes[mask].clone()
        c_scores = scores[mask].clone()
        
        while c_boxes.shape[0] > 0:
            # Index of maximum score box
            max_idx = torch.argmax(c_scores)
            
            best_box = c_boxes[max_idx]
            best_score = c_scores[max_idx]
            
            # Save the box if it meets the confidence threshold
            if best_score > confidence_threshold:
                keep_boxes.append(best_box)
                keep_scores.append(best_score)
                keep_labels.append(c)
            
            # Remove the selected box from active search list
            c_boxes = torch.cat([c_boxes[:max_idx], c_boxes[max_idx + 1:]], dim=0)
            c_scores = torch.cat([c_scores[:max_idx], c_scores[max_idx + 1:]], dim=0)
            
            if c_boxes.shape[0] == 0:
                break
                
            # Compute IoU of best box with remaining boxes
            ious = bbox_iou_tensor(c_boxes, best_box.unsqueeze(0)).squeeze(1)
            
            # Apply Gaussian decay to scores of overlapping boxes
            decay = torch.exp(-(ious ** 2) / sigma)
            c_scores = c_scores * decay
            
            # Remove boxes whose scores dropped below the threshold
            keep_mask = c_scores > score_threshold
            c_boxes = c_boxes[keep_mask]
            c_scores = c_scores[keep_mask]
            
    if len(keep_boxes) > 0:
        return torch.stack(keep_boxes), torch.stack(keep_scores), torch.stack(keep_labels)
    else:
        device = boxes.device
        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), device=device),
            torch.empty((0,), dtype=torch.long, device=device)
        )
