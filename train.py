import os
import sys
import argparse
import json
import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add public/tools to path to import evaluate_predictions
sys.path.append(os.path.abspath("public/tools"))
try:
    from evaluate_predictions import evaluate, validate_ground_truth, normalize_predictions
except ImportError:
    # Fallback if evaluated script is moved
    evaluate = None

from models.fcos import FCOS
from utils.dataset import DetectionDataset, collate_fn, letterbox
from utils.assign import FCOSLabelAssigner
from utils.loss import FCOSWithCIoULoss
from utils.nms import soft_nms

WARMUP_EPOCHS = 3
MAIN_END_EPOCH = 28
DEFAULT_EPOCHS = 35
HEAD_MAX_LR = 1e-3
BACKBONE_MAX_LR = HEAD_MAX_LR * 0.1
FINE_TUNE_LR = 5e-6

def parse_args():
    parser = argparse.ArgumentParser(description="Train custom FCOS detector.")
    parser.add_argument("--train_data", default="./public/annotations/train.json")
    parser.add_argument("--val_data", default="./public/annotations/val.json")
    parser.add_argument("--image_dir", default="./public/train/images")
    parser.add_argument("--val_image_dir", default="./public/val/images")
    parser.add_argument("--checkpoint_dir", default="./models/")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Stop training if validation mAP does not improve for this many epochs. Use 0 to disable.")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                        help="Minimum mAP improvement required to reset early stopping patience.")
    parser.add_argument("--min_epochs", type=int, default=25,
                        help="Minimum number of epochs to run before early stopping can trigger.")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="Final EMA decay. A warmup schedule is applied automatically at the start.")
    parser.add_argument("--plateau_switch_patience", type=int, default=3,
                        help="Switch to fine-tune early if mAP improvement stays below the plateau delta for this many validations.")
    parser.add_argument("--plateau_switch_min_delta", type=float, default=0.005,
                        help="Minimum mAP improvement per epoch considered meaningful for main training.")
    parser.add_argument("--earliest_fine_tune_epoch", type=int, default=13,
                        help="Earliest epoch index where plateau-based fine-tune switching is allowed.")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.tau = tau
        self.updates = 0
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            decay = self.decay * (1.0 - math.exp(-self.updates / self.tau))
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in msd:
                if esd[k].dtype.is_floating_point:
                    esd[k].mul_(decay).add_(msd[k].detach(), alpha=1.0 - decay)

def set_phase_learning_rate(optimizer, epoch, fine_tune_epoch):
    if epoch < WARMUP_EPOCHS:
        lr_head = HEAD_MAX_LR * (epoch + 1) / WARMUP_EPOCHS
        lr_backbone = 0.0
    elif epoch >= fine_tune_epoch:
        lr_head = FINE_TUNE_LR
        lr_backbone = 0.0
    else:
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        return lr_head, lr_backbone

    optimizer.param_groups[0]["lr"] = lr_backbone
    optimizer.param_groups[1]["lr"] = lr_head
    return lr_head, lr_backbone

def configure_backbone_gradients(model, epoch, fine_tune_epoch):
    if epoch < WARMUP_EPOCHS or epoch >= fine_tune_epoch:
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True

def evaluate_model(model, dataloader, assigner, classes, gt_data, val_image_info, device):
    model.eval()
    predictions_list = []
    
    # Pre-get location strides
    locations = assigner.locations.to(device)
    strides = assigner.loc_strides.to(device)

    with torch.no_grad():
        for images, _, _, img_ids, orig_sizes in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            cls_logits = outputs["cls_logits"]
            bbox_preds = outputs["bbox_preds"]
            centerness_logits = outputs["centerness_logits"]
            
            # Decode for each image in batch
            batch_size = images.shape[0]
            for b in range(batch_size):
                img_id = img_ids[b]
                orig_w, orig_h = orig_sizes[b]
                
                # Extract image predictions
                img_cls_logits = cls_logits[b]       # M x C
                img_bbox_preds = bbox_preds[b]       # M x 4 (l, t, r, b)
                img_centerness_logits = centerness_logits[b] # M
                
                # Apply sigmoid to get scores
                scores = torch.sigmoid(img_cls_logits) * torch.sigmoid(img_centerness_logits)[:, None]
                max_scores, class_ids = scores.max(dim=1)
                
                # Keep confidence > 0.05
                keep = max_scores > 0.05
                if keep.sum() == 0:
                    predictions_list.append({"image_id": img_id, "boxes": []})
                    continue
                    
                c_bbox = img_bbox_preds[keep]
                c_scores = max_scores[keep]
                c_class_ids = class_ids[keep]
                c_locations = locations[keep]
                
                # Decode offsets [l, t, r, b] to [xmin, ymin, xmax, ymax]
                xmin = c_locations[:, 0] - c_bbox[:, 0]
                ymin = c_locations[:, 1] - c_bbox[:, 1]
                xmax = c_locations[:, 0] + c_bbox[:, 2]
                ymax = c_locations[:, 1] + c_bbox[:, 3]
                
                boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
                
                # Map back to original image size (Letterbox inverse)
                # Stride scale
                r = min(assigner.img_size / orig_h, assigner.img_size / orig_w)
                left = (assigner.img_size - orig_w * r) / 2.0
                top = (assigner.img_size - orig_h * r) / 2.0
                
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
                
                # Clip to image bounds
                boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0, max=orig_w)
                boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0, max=orig_h)
                valid_boxes = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
                boxes = boxes[valid_boxes]
                c_scores = c_scores[valid_boxes]
                c_class_ids = c_class_ids[valid_boxes]
                if boxes.shape[0] == 0:
                    predictions_list.append({"image_id": img_id, "boxes": []})
                    continue
                
                # Apply class-wise Soft-NMS
                keep_boxes, keep_scores, keep_labels = soft_nms(
                    boxes, c_scores, c_class_ids, sigma=0.5, score_threshold=0.001, confidence_threshold=0.05
                )
                if keep_scores.numel() > 0:
                    order = torch.argsort(keep_scores, descending=True)
                    keep_boxes = keep_boxes[order]
                    keep_scores = keep_scores[order]
                    keep_labels = keep_labels[order]
                
                boxes_list = []
                for i in range(keep_boxes.shape[0]):
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
                    
                predictions_list.append({
                    "image_id": img_id,
                    "boxes": boxes_list
                })

    # Evaluate predictions list
    if evaluate is not None:
        try:
            normalized_preds = normalize_predictions(
                predictions_list,
                classes=classes,
                image_info=val_image_info,
                max_detections_per_image=100,
                require_complete=False
            )
            eval_results = evaluate(
                ground_truth=gt_data,
                predictions=normalized_preds,
                classes=classes,
                iou_threshold=0.5
            )
            max_conf = max((pred["confidence"] for pred in normalized_preds), default=0.0)
            print(
                f"Validation predictions: {len(normalized_preds)} boxes, "
                f"max_conf={max_conf:.4f}, "
                f"micro_recall={eval_results.get('micro_recall', 0.0):.4f}, "
                f"micro_precision={eval_results.get('micro_precision', 0.0):.4f}"
            )
            return eval_results["mAP@0.5"]
        except Exception as e:
            print(f"Error in evaluate evaluation: {e}")
            return 0.0
    else:
        print("evaluate_predictions.py evaluate function is not available.")
        return 0.0

def main():
    args = parse_args()
    set_seed(args.seed)

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build datasets
    train_dataset = DetectionDataset(args.train_data, args.image_dir, img_size=args.img_size, augment=True)
    val_dataset = DetectionDataset(args.val_data, args.val_image_dir, img_size=args.img_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    classes = train_dataset.classes
    print(f"Classes: {classes}")

    # For validation helper
    if evaluate is not None:
        with open(args.val_data, "r", encoding="utf-8-sig") as f:
            gt_data = json.load(f)
        _, val_image_info = validate_ground_truth(gt_data)
    else:
        gt_data, val_image_info = None, None

    # Load Model
    model = FCOS(num_classes=len(classes), pretrained=True).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)

    # Label assigner & Loss
    assigner = FCOSLabelAssigner(img_size=args.img_size)
    criterion = FCOSWithCIoULoss()

    # Optimizer configuration
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.fpn.parameters()) + list(model.head.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 0.0},
        {"params": head_params, "lr": HEAD_MAX_LR}
    ], weight_decay=1e-4)

    main_epochs = max(0, min(args.epochs, MAIN_END_EPOCH) - WARMUP_EPOCHS)
    onecycle_scheduler = None
    if main_epochs > 0:
        onecycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[BACKBONE_MAX_LR, HEAD_MAX_LR],
            total_steps=main_epochs * len(train_loader),
            pct_start=0.15,
            anneal_strategy="cos",
            div_factor=1.0,
            final_div_factor=200.0
        )

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_map = -float("inf")
    epochs_without_improvement = 0
    fine_tune_epoch = min(MAIN_END_EPOCH, args.epochs)
    plateau_epochs = 0
    previous_val_map = None

    print("Starting training...")
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch, warmup_epochs=WARMUP_EPOCHS, fine_tune_epoch=fine_tune_epoch)
        # 1. Adjust gradients and LR
        configure_backbone_gradients(model, epoch, fine_tune_epoch)
        lr_head, lr_backbone = set_phase_learning_rate(optimizer, epoch, fine_tune_epoch)
        if epoch == WARMUP_EPOCHS:
            print("Starting main training phase: unfreezing backbone and enabling OneCycleLR.")
            optimizer.param_groups[0]["lr"] = BACKBONE_MAX_LR
            optimizer.param_groups[1]["lr"] = HEAD_MAX_LR
        if epoch == fine_tune_epoch:
            print("Starting fine-tune phase: freezing backbone and disabling Mosaic/Random Scale.")

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_ctr_loss = 0.0
        
        for batch_idx, (images, bboxes, labels, _, _) in enumerate(pbar):
            images = images.to(device)
            # Move ground truth boxes and labels to device
            bboxes = [b.to(device) for b in bboxes]
            labels = [l.to(device) for l in labels]

            # Assign labels
            cls_targets, reg_targets, centerness_targets = assigner(bboxes, labels)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(images)

            # Compute loss
            loss_dict = criterion(predictions, cls_targets, reg_targets, centerness_targets)
            loss = loss_dict["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Nan/Inf loss encountered in batch {batch_idx}. Skipping step.")
                continue

            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            if onecycle_scheduler is not None and WARMUP_EPOCHS <= epoch < fine_tune_epoch:
                onecycle_scheduler.step()

            # Update EMA weights
            ema.update(model)

            epoch_loss += loss.item()
            epoch_cls_loss += loss_dict["loss_cls"].item()
            epoch_reg_loss += loss_dict["loss_reg"].item()
            epoch_ctr_loss += loss_dict["loss_centerness"].item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_dict['loss_cls'].item():.4f}",
                "reg": f"{loss_dict['loss_reg'].item():.4f}",
                "pos": int(loss_dict["num_pos"])
            })

        avg_loss = epoch_loss / len(train_loader)
        avg_cls_loss = epoch_cls_loss / len(train_loader)
        avg_reg_loss = epoch_reg_loss / len(train_loader)
        avg_ctr_loss = epoch_ctr_loss / len(train_loader)

        print(f"Epoch {epoch} finished. Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Reg: {avg_reg_loss:.4f}, Ctr: {avg_ctr_loss:.4f})")
        print(f"Learning rates: Backbone={optimizer.param_groups[0]['lr']:.2e}, Head={optimizer.param_groups[1]['lr']:.2e}")

        # Run Validation Evaluation on EMA model every epoch
        val_map = evaluate_model(ema.ema, val_loader, assigner, classes, gt_data, val_image_info, device)
        print(f"Validation mAP@0.5: {val_map:.4f}")

        if WARMUP_EPOCHS <= epoch < fine_tune_epoch and previous_val_map is not None:
            map_delta = val_map - previous_val_map
            if map_delta < args.plateau_switch_min_delta:
                plateau_epochs += 1
            else:
                plateau_epochs = 0
            if (
                args.plateau_switch_patience > 0
                and epoch >= args.earliest_fine_tune_epoch
                and plateau_epochs >= args.plateau_switch_patience
            ):
                fine_tune_epoch = epoch + 1
                plateau_epochs = 0
                print(
                    f"Plateau detected: mAP delta < {args.plateau_switch_min_delta:.4f} "
                    f"for {args.plateau_switch_patience} validation(s). "
                    f"Switching to fine-tune at epoch {fine_tune_epoch}."
                )
        previous_val_map = val_map

        # Save checkpoint
        improved = val_map > best_map + args.early_stop_min_delta
        if improved:
            best_map = val_map
            epochs_without_improvement = 0
            print(f"mAP improved! Saving best weights to {args.checkpoint_dir}best.pth")
            torch.save(ema.ema.state_dict(), os.path.join(args.checkpoint_dir, "best.pth"))
        else:
            epochs_without_improvement += 1
            print(
                f"No mAP improvement for {epochs_without_improvement} epoch(s). "
                f"Best mAP@0.5: {best_map:.4f}"
            )
            
        # Also save last model weights for resuming
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.ema.state_dict(),
            "ema_updates": ema.updates,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_map": best_map,
            "fine_tune_epoch": fine_tune_epoch
        }, os.path.join(args.checkpoint_dir, "last.pth"))

        if (
            args.early_stop_patience > 0
            and epoch + 1 >= args.min_epochs
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best Validation mAP@0.5: {best_map:.4f}"
            )
            break

    print(f"Training complete. Best Validation mAP@0.5: {best_map:.4f}")

if __name__ == "__main__":
    main()
