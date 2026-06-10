import torch

class FCOSLabelAssigner:
    def __init__(self, img_size=416, strides=[8, 16, 32], limit_ranges=[[0, 64], [64, 128], [128, 1000000]]):
        self.img_size = img_size
        self.strides = strides
        self.limit_ranges = limit_ranges
        
        # Precompute locations, level_limits, and strides for a given image size
        locations, limits, loc_strides = self._compute_locations()
        self.register_buffer('locations', locations)
        self.register_buffer('limits', limits)
        self.register_buffer('loc_strides', loc_strides)

    def register_buffer(self, name, tensor):
        # Helper to simulate PyTorch Module buffer registration behavior
        setattr(self, name, tensor)

    def _compute_locations(self):
        locations = []
        limits = []
        loc_strides = []
        
        for stride, limit in zip(self.strides, self.limit_ranges):
            h_size = self.img_size // stride
            w_size = self.img_size // stride
            
            # Grid of coordinates
            shift_x = (torch.arange(0, w_size, dtype=torch.float32) + 0.5) * stride
            shift_y = (torch.arange(0, h_size, dtype=torch.float32) + 0.5) * stride
            
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            
            locs = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1) # shape (H*W, 2)
            locations.append(locs)
            
            lims = locs.new_tensor([limit]).repeat(locs.shape[0], 1) # shape (H*W, 2)
            limits.append(lims)
            
            strids = locs.new_tensor([stride]).repeat(locs.shape[0]) # shape (H*W)
            loc_strides.append(strids)
            
        locations = torch.cat(locations, dim=0)
        limits = torch.cat(limits, dim=0)
        loc_strides = torch.cat(loc_strides, dim=0)
        return locations, limits, loc_strides

    def to(self, device):
        self.locations = self.locations.to(device)
        self.limits = self.limits.to(device)
        self.loc_strides = self.loc_strides.to(device)
        return self

    def __call__(self, gt_bboxes_list, gt_labels_list):
        # Match device
        device = gt_bboxes_list[0].device if len(gt_bboxes_list) > 0 else torch.device('cpu')
        self.to(device)
        
        batch_size = len(gt_bboxes_list)
        num_locs = self.locations.shape[0]
        
        batch_cls_targets = []
        batch_reg_targets = []
        batch_centerness_targets = []
        
        for b in range(batch_size):
            bboxes = gt_bboxes_list[b]  # shape (N, 4)
            labels = gt_labels_list[b]  # shape (N,)
            
            if len(bboxes) == 0:
                # No objects in this image
                batch_cls_targets.append(self.locations.new_full((num_locs,), -1, dtype=torch.long))
                batch_reg_targets.append(self.locations.new_zeros((num_locs, 4)))
                batch_centerness_targets.append(self.locations.new_zeros((num_locs,)))
                continue
            
            # Compute distances from locations to all bounding boxes
            # Locations shape: (M, 2)
            # Bboxes shape: (N, 4) -> [xmin, ymin, xmax, ymax]
            xs, ys = self.locations[:, 0], self.locations[:, 1] # shape (M,)
            
            # Calculate l, t, r, b for all locations and boxes
            l = xs[:, None] - bboxes[None, :, 0] # shape (M, N)
            t = ys[:, None] - bboxes[None, :, 1] # shape (M, N)
            r = bboxes[None, :, 2] - xs[:, None] # shape (M, N)
            b = bboxes[None, :, 3] - ys[:, None] # shape (M, N)
            
            reg_targets = torch.stack([l, t, r, b], dim=2) # shape (M, N, 4)
            
            # Condition 1: Location must be inside the bounding box
            is_in_box = reg_targets.min(dim=2)[0] > 0 # shape (M, N)
            
            # Condition 2: Max distance must fall inside the range limit of the level
            max_reg = reg_targets.max(dim=2)[0] # shape (M, N)
            is_in_scale = (max_reg >= self.limits[:, 0, None]) & (max_reg <= self.limits[:, 1, None])
            
            # Condition 3: Center sampling (pixel must be close to center)
            # Center coordinates
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
            
            radius = 1.5 * self.loc_strides # shape (M,)
            
            c_xmin = torch.max(bboxes[:, 0], cx - radius[:, None])
            c_ymin = torch.max(bboxes[:, 1], cy - radius[:, None])
            c_xmax = torch.min(bboxes[:, 2], cx + radius[:, None])
            c_ymax = torch.min(bboxes[:, 3], cy + radius[:, None])
            
            cl = xs[:, None] - c_xmin
            ct = ys[:, None] - c_ymin
            cr = c_xmax - xs[:, None]
            cb = c_ymax - ys[:, None]
            
            c_reg_targets = torch.stack([cl, ct, cr, cb], dim=2)
            is_in_center = c_reg_targets.min(dim=2)[0] > 0 # shape (M, N)
            
            # Combine all conditions
            is_pos = is_in_box & is_in_scale & is_in_center # shape (M, N)
            
            # Calculate areas for ambiguity resolution
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) # shape (N,)
            areas_mask = areas[None, :].repeat(num_locs, 1) # shape (M, N)
            areas_mask[~is_pos] = 1e18 # Set invalid to infinity
            
            # Select box with smallest area
            min_areas, best_box_idx = areas_mask.min(dim=1) # shape (M,)
            
            # Positive locations are those where min_area is less than 1e18
            is_pos_final = min_areas < 1e18 # shape (M,)
            
            # Generate targets
            cls_t = labels[best_box_idx] # shape (M,)
            cls_t[~is_pos_final] = -1 # Background
            
            # Bbox offsets for positive locations
            reg_t = reg_targets[torch.arange(num_locs, device=device), best_box_idx] # shape (M, 4)
            # Set non-positive to 0
            reg_t[~is_pos_final] = 0.0
            
            # Centerness targets
            best_l, best_t, best_r, best_b = reg_t[:, 0], reg_t[:, 1], reg_t[:, 2], reg_t[:, 3]
            min_lr = torch.min(best_l, best_r)
            max_lr = torch.max(best_l, best_r)
            min_tb = torch.min(best_t, best_b)
            max_tb = torch.max(best_t, best_b)
            
            eps = 1e-6
            centerness_t = torch.sqrt((min_lr / (max_lr + eps)) * (min_tb / (max_tb + eps)))
            centerness_t[~is_pos_final] = 0.0
            
            batch_cls_targets.append(cls_t)
            batch_reg_targets.append(reg_t)
            batch_centerness_targets.append(centerness_t)
            
        return (
            torch.stack(batch_cls_targets, dim=0),       # B x M
            torch.stack(batch_reg_targets, dim=0),       # B x M x 4
            torch.stack(batch_centerness_targets, dim=0) # B x M
        )
