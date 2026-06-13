import os
import json
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.fcos import FCOS
from utils.dataset import letterbox
from utils.assign import FCOSLabelAssigner
from utils.nms import soft_nms

def parse_args():
    parser = argparse.ArgumentParser(description="Predict using custom FCOS detector.")
    parser.add_argument("--image_dir", required=True, help="Directory containing images to predict.")
    parser.add_argument("--output", required=True, help="Path to write output predictions JSON.")
    parser.add_argument("--checkpoint", default="./models/best.pth", help="Path to model weights checkpoint.")
    parser.add_argument("--classes_file", default="./public/classes.json", help="Path to classes.json file.")
    parser.add_argument("--img_size", type=int, default=416, help="Model input size.")
    parser.add_argument("--conf_thresh", type=float, default=0.05, help="Confidence threshold for predictions.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load classes
    with open(args.classes_file, "r", encoding="utf-8") as f:
        classes = json.load(f)
    print(f"Loaded classes: {classes}")

    # Instantiate model
    model = FCOS(num_classes=len(classes), pretrained=False)
    
    # Load weights
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
    
    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    # If saved as full checkpoint dict, extract weights
    if "model_state_dict" in state_dict:
        state_dict = state_dict["ema_state_dict"] if "ema_state_dict" in state_dict else state_dict["model_state_dict"]
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Pre-generate FCOS location coordinates
    assigner = FCOSLabelAssigner(img_size=args.img_size)
    locations = assigner.locations.to(device)

    # Scan image directory
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [
        f for f in os.listdir(args.image_dir) 
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    print(f"Found {len(img_files)} images for prediction.")

    predictions = []

    # Process images one by one (or in batches, one by one is simple and robust)
    with torch.no_grad():
        for filename in tqdm(img_files, desc="Inference"):
            img_path = os.path.join(args.image_dir, filename)
            img_raw = cv2.imread(img_path)
            if img_raw is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                predictions.append({
                    "image_id": filename,
                    "boxes": []
                })
                continue
                
            orig_h, orig_w = img_raw.shape[:2]
            
            # Apply letterbox preprocessing
            img, r, (left, top) = letterbox(img_raw, new_shape=args.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            
            # Convert to tensor and batch dimension (1, 3, H, W)
            img = np.transpose(img, (2, 0, 1))
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

            # Inference
            outputs = model(img_tensor)
            
            cls_logits = outputs["cls_logits"][0]       # M x C
            bbox_preds = outputs["bbox_preds"][0]       # M x 4 (l, t, r, b)
            centerness_logits = outputs["centerness_logits"][0] # M
            has_centerness = outputs.get("has_centerness", True)
            
            # Compute classification scores
            if has_centerness:
                scores = torch.sigmoid(cls_logits) * torch.sigmoid(centerness_logits)[:, None]
            else:
                scores = torch.sigmoid(cls_logits)
            max_scores, class_ids = scores.max(dim=1)
            
            # Keep predictions above confidence threshold
            keep = max_scores > args.conf_thresh
            if keep.sum() == 0:
                predictions.append({
                    "image_id": filename,
                    "boxes": []
                })
                continue
                
            c_bbox = bbox_preds[keep]
            c_scores = max_scores[keep]
            c_class_ids = class_ids[keep]
            c_locations = locations[keep]
            
            # Decode offsets to box coordinates
            xmin = c_locations[:, 0] - c_bbox[:, 0]
            ymin = c_locations[:, 1] - c_bbox[:, 1]
            xmax = c_locations[:, 0] + c_bbox[:, 2]
            ymax = c_locations[:, 1] + c_bbox[:, 3]
            
            boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
            
            # Convert coordinates back to original image size (Letterbox inverse)
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
            
            # Clip coordinates to original image bounds
            boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0, max=orig_w)
            boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0, max=orig_h)
            valid_boxes = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
            boxes = boxes[valid_boxes]
            c_scores = c_scores[valid_boxes]
            c_class_ids = c_class_ids[valid_boxes]
            if boxes.shape[0] == 0:
                predictions.append({
                    "image_id": filename,
                    "boxes": []
                })
                continue
            
            # Apply class-wise Soft-NMS
            keep_boxes, keep_scores, keep_labels = soft_nms(
                boxes, c_scores, c_class_ids, sigma=0.5, score_threshold=0.001, confidence_threshold=args.conf_thresh
            )
            if keep_scores.numel() > 0:
                order = torch.argsort(keep_scores, descending=True)
                keep_boxes = keep_boxes[order]
                keep_scores = keep_scores[order]
                keep_labels = keep_labels[order]
            
            # Keep at most 100 predictions per image (as per evaluate_predictions.py default)
            num_keep = min(keep_boxes.shape[0], 100)
            
            boxes_list = []
            for i in range(num_keep):
                cls_idx = keep_labels[i].item()
                boxes_list.append({
                    "class": classes[cls_idx],
                    "confidence": round(keep_scores[i].item(), 4),
                    "bbox": [
                        round(keep_boxes[i, 0].item(), 2),
                        round(keep_boxes[i, 1].item(), 2),
                        round(keep_boxes[i, 2].item(), 2),
                        round(keep_boxes[i, 3].item(), 2)
                    ]
                })
                
            predictions.append({
                "image_id": filename,
                "boxes": boxes_list
            })

    # Save to JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
