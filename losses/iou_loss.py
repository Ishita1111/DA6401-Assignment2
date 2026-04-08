"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # Convert (x_center, y_center, w, h) → (x1, y1, x2, y2)

        px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        tx, ty, tw, th = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

        p_x1 = px - pw / 2
        p_y1 = py - ph / 2
        p_x2 = px + pw / 2
        p_y2 = py + ph / 2

        t_x1 = tx - tw / 2
        t_y1 = ty - th / 2
        t_x2 = tx + tw / 2
        t_y2 = ty + th / 2

        # Intersection
        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)

        inter_area = inter_w * inter_h

        # Areas
        pred_area = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
        target_area = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)

        # Union
        union = pred_area + target_area - inter_area + self.eps

        # IoU
        iou = inter_area / union

        # Loss
        loss = 1 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss