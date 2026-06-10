import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def clip_and_filter_boxes(bboxes, labels, img_size, min_size=2.0):
    if len(bboxes) == 0:
        return bboxes, labels
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img_size)
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img_size)
    valid = (bboxes[:, 2] > bboxes[:, 0] + min_size) & (bboxes[:, 3] > bboxes[:, 1] + min_size)
    return bboxes[valid], labels[valid]

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114)):
    # Resize and pad image while preserving aspect ratio
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (left, top)

class DetectionDataset(Dataset):
    def __init__(self, annotation_path, image_dir, img_size=416, augment=False):
        self.image_dir = image_dir
        self.img_size = img_size
        self.augment = augment
        self.mosaic_enabled = True
        self.mosaic_prob = 0.5

        # Load annotations
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.classes = data["classes"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Build image annotations dictionary
        self.img_ann = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_ann:
                self.img_ann[img_id] = []
            self.img_ann[img_id].append(ann)

        self.images = data["images"]
        # Filter out images that do not exist (optional, safety check)
        self.images = [img for img in self.images if os.path.exists(os.path.join(self.image_dir, img["file_name"].split("/")[-1]))]

    def __len__(self):
        return len(self.images)

    def set_epoch(self, epoch):
        if not self.augment:
            return
        if epoch >= 85:
            self.mosaic_enabled = False
            return
        # Ramp mosaic probability from 0.5 to 0.8 during the main training stage.
        self.mosaic_prob = min(0.8, 0.5 + 0.3 * (epoch / 84.0))

    def load_image_and_boxes(self, index):
        img_info = self.images[index]
        img_id = img_info["id"]
        # Use only basename to find the image in the current images directory
        basename = img_info["file_name"].split("/")[-1]
        img_path = os.path.join(self.image_dir, basename)
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        anns = self.img_ann.get(img_id, [])
        bboxes = []
        labels = []
        for ann in anns:
            class_name = ann["class"]
            class_idx = self.class_to_idx[class_name]
            # Coordinates are [xmin, ymin, xmax, ymax]
            bbox = list(ann["bbox"])
            bboxes.append(bbox)
            labels.append(class_idx)

        return img, np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64), (w, h), img_id

    def load_mosaic(self, index):
        # 2x2 mosaic augmentation
        bboxes4 = []
        labels4 = []
        
        # Center of the mosaic image
        s = self.img_size
        xc, yc = [int(random.uniform(s // 2, 2 * s - s // 2)) for _ in range(2)]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.images) - 1) for _ in range(3)]
        
        # Create a black canvas
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)

        for i, idx in enumerate(indices):
            # Load image
            img, bboxes, labels, _, _ = self.load_image_and_boxes(idx)
            h, w = img.shape[:2]

            # Place image in quadrant
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = 0, 0, xc, yc
                x1b, y1b, x2b, y2b = w - xc, h - yc, w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, 0, s * 2, yc
                x1b, y1b, x2b, y2b = 0, h - yc, min(w, s * 2 - xc), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = 0, yc, xc, s * 2
                x1b, y1b, x2b, y2b = w - xc, 0, w, min(h, s * 2 - yc)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, s * 2, s * 2
                x1b, y1b, x2b, y2b = 0, 0, min(w, s * 2 - xc), min(h, s * 2 - yc)

            # Ensure coordinates are within valid range (clip if necessary)
            # Clip src coords
            x1b = max(0, x1b)
            y1b = max(0, y1b)
            x2b = min(w, x2b)
            y2b = min(h, y2b)
            
            # Clip dst coords based on actual sliced src coords size
            w_slice = x2b - x1b
            h_slice = y2b - y1b
            if i == 0:
                x1a = xc - w_slice
                y1a = yc - h_slice
            elif i == 1:
                x2a = xc + w_slice
                y1a = yc - h_slice
            elif i == 2:
                x1a = xc - w_slice
                y2a = yc + h_slice
            elif i == 3:
                x2a = xc + w_slice
                y2a = yc + h_slice

            # Copy slice into canvas
            if w_slice > 0 and h_slice > 0:
                mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            # Adjust bounding boxes
            padw = x1a - x1b
            padh = y1a - y1b
            if len(bboxes) > 0:
                bboxes_quad = bboxes.copy()
                bboxes_quad[:, [0, 2]] = bboxes_quad[:, [0, 2]] + padw
                bboxes_quad[:, [1, 3]] = bboxes_quad[:, [1, 3]] + padh
                # Clip boxes to quadrant boundary
                bboxes_quad[:, [0, 2]] = np.clip(bboxes_quad[:, [0, 2]], x1a, x2a)
                bboxes_quad[:, [1, 3]] = np.clip(bboxes_quad[:, [1, 3]], y1a, y2a)
                bboxes_quad, labels_quad = clip_and_filter_boxes(bboxes_quad, labels, s * 2)
                bboxes4.append(bboxes_quad)
                labels4.append(labels_quad)

        if len(bboxes4) > 0:
            bboxes4 = np.concatenate(bboxes4, axis=0)
            labels4 = np.concatenate(labels4, axis=0)
        else:
            bboxes4 = np.empty((0, 4), dtype=np.float32)
            labels4 = np.empty((0,), dtype=np.int64)

        # Crop / Resize mosaic image to s x s
        # Crop a random s x s block that contains some parts of the quadrants
        # We can crop around the center (xc, yc) to ensure we get a mix
        c_x1 = max(0, xc - s // 2)
        c_y1 = max(0, yc - s // 2)
        c_x2 = min(s * 2, c_x1 + s)
        c_y2 = min(s * 2, c_y1 + s)
        
        # If boundary conditions made crop size smaller than s, shift back
        if c_x2 - c_x1 < s:
            c_x1 = max(0, c_x2 - s)
        if c_y2 - c_y1 < s:
            c_y1 = max(0, c_y2 - s)

        cropped_img = mosaic_img[c_y1:c_y1+s, c_x1:c_x1+s]
        
        # Adjust boxes to cropped coordinates
        if len(bboxes4) > 0:
            bboxes4[:, [0, 2]] = bboxes4[:, [0, 2]] - c_x1
            bboxes4[:, [1, 3]] = bboxes4[:, [1, 3]] - c_y1
            bboxes4, labels4 = clip_and_filter_boxes(bboxes4, labels4, s)

        return cropped_img, bboxes4, labels4

    def random_scale_letterbox(self, img, bboxes):
        scale = random.uniform(0.5, 1.5)
        h, w = img.shape[:2]
        scaled_w = max(1, int(round(w * scale)))
        scaled_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        if len(bboxes) > 0:
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] *= scaled_w / w
            bboxes[:, [1, 3]] *= scaled_h / h

        if scaled_w >= self.img_size and scaled_h >= self.img_size:
            max_x = scaled_w - self.img_size
            max_y = scaled_h - self.img_size
            x0 = random.randint(0, max_x) if max_x > 0 else 0
            y0 = random.randint(0, max_y) if max_y > 0 else 0
            img = img[y0:y0 + self.img_size, x0:x0 + self.img_size]
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] -= x0
                bboxes[:, [1, 3]] -= y0
            left, top = 0, 0
        else:
            canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
            left = random.randint(0, self.img_size - scaled_w) if scaled_w < self.img_size else 0
            top = random.randint(0, self.img_size - scaled_h) if scaled_h < self.img_size else 0
            crop = img[:self.img_size, :self.img_size]
            ch, cw = crop.shape[:2]
            canvas[top:top + ch, left:left + cw] = crop
            img = canvas
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] += left
                bboxes[:, [1, 3]] += top

        return img, bboxes

    def __getitem__(self, index):
        if self.augment and self.mosaic_enabled and random.random() < self.mosaic_prob:
            # Load Mosaic
            img, bboxes, labels = self.load_mosaic(index)
            # Original shape is considered to be img_size since it's synthetic
            orig_size = (self.img_size, self.img_size)
            img_info = self.images[index]
            img_id = img_info["id"]
        else:
            # Load standard image with letterboxing
            img, bboxes, labels, orig_size, img_id = self.load_image_and_boxes(index)
            if self.augment:
                img, bboxes = self.random_scale_letterbox(img, bboxes)
            else:
                # Apply letterbox
                img, r, (left, top) = letterbox(img, new_shape=self.img_size)
                if len(bboxes) > 0:
                    # Update box coordinates according to letterboxing
                    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * r + left
                    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * r + top

            bboxes, labels = clip_and_filter_boxes(bboxes, labels, self.img_size)

        # Apply standard augmentations
        if self.augment:
            # Random Horizontal Flip
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                if len(bboxes) > 0:
                    xmin_new = self.img_size - bboxes[:, 2]
                    xmax_new = self.img_size - bboxes[:, 0]
                    bboxes[:, 0] = xmin_new
                    bboxes[:, 2] = xmax_new

            # Color Jitter (simple implementation in numpy)
            if random.random() < 0.5:
                # Brightness
                factor = random.uniform(0.6, 1.4)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)
                
            if random.random() < 0.5:
                # Contrast
                factor = random.uniform(0.6, 1.4)
                mean = img.mean(axis=(0, 1), keepdims=True)
                img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

            if random.random() < 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] *= random.uniform(0.6, 1.4)
                hsv[:, :, 0] += random.uniform(-18.0, 18.0)
                hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180.0)
                hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            if random.random() < 0.1:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Normalize and convert to tensor
        img = img.astype(np.float32) / 255.0
        # ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Convert HWC to CHW
        img = np.transpose(img, (2, 0, 1))

        # Convert to tensors
        img_tensor = torch.from_numpy(img)
        bboxes_tensor = torch.from_numpy(bboxes)
        labels_tensor = torch.from_numpy(labels)

        return img_tensor, bboxes_tensor, labels_tensor, img_id, orig_size

def collate_fn(batch):
    images, bboxes, labels, img_ids, orig_sizes = zip(*batch)
    
    images = torch.stack(images, dim=0)
    # Bboxes and labels are left as lists of tensors since they vary in size
    return images, list(bboxes), list(labels), list(img_ids), list(orig_sizes)
