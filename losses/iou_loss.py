"""Custom IoU loss for bounding box regression.

Computes 1 − IoU so that the loss is in [0, 1].
Boxes are in (x_center, y_center, width, height) pixel format.
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Loss = 1 − IoU ∈ [0, 1].
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.

        Args:
            eps: Small value to avoid division by zero.
            reduction: 'mean' | 'sum' | 'none'.
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}'. Must be 'mean', 'sum', or 'none'.")
        self.eps = eps
        self.reduction = reduction

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes.unbind(dim=-1)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format.

        Returns:
            Scalar loss (if reduction='mean'|'sum') or per-sample loss [B] (if 'none').
        """
        # Convert to corner format
        pred_xyxy = self._cxcywh_to_xyxy(pred_boxes)
        target_xyxy = self._cxcywh_to_xyxy(target_boxes)

        # Intersection rectangle
        inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
        inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
        inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
        inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Areas of each box
        pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * \
                    (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
        target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0) * \
                      (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=0)

        union_area = pred_area + target_area - inter_area

        # IoU with numerical stability
        iou = inter_area / (union_area + self.eps)

        # Loss = 1 - IoU  (in [0, 1])
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss