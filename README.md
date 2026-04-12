# DA6401 Assignment 2 — Multi-Task Visual Perception

**W&B Report:** [https://api.wandb.ai/links/da25m020-indian-institute-of-technology-madras/y4rwjl32](https://api.wandb.ai/links/da25m020-indian-institute-of-technology-madras/y4rwjl32)

**GitHub Repo:** [https://github.com/usnaveen/A2_Deep_Learning](https://github.com/usnaveen/A2_Deep_Learning)

## Overview

End-to-end visual perception pipeline on the Oxford-IIIT Pet dataset using a shared VGG11-BN encoder with three task heads:

1. **Classification** — 37-breed classification (F1 >= 0.93)
2. **Localization** — Bounding box regression (Acc@IoU0.5 >= 91%)
3. **Segmentation** — Trimap segmentation via U-Net decoder (Dice >= 0.82)

All three heads share a single VGG11-BN encoder. Model checkpoints are automatically downloaded from Google Drive at inference time.

## Project Structure

```
├── models/
│   ├── vgg11.py             # Shared VGG11-BN encoder
│   ├── classification.py    # Breed classification head
│   ├── localization.py      # Bounding box regression head
│   ├── segmentation.py      # U-Net decoder for trimap segmentation
│   ├── multitask.py         # Shared-backbone multi-task model
│   └── layers.py            # CustomDropout
├── losses/
│   └── iou_loss.py          # Custom IoU loss
├── data/
│   └── pets_dataset.py      # Oxford-IIIT Pet dataset + transforms
├── train.py                 # Training loop for all three tasks
├── inference.py             # Inference and evaluation utilities
├── multitask.py             # Entry point (imports MultiTaskPerceptionModel)
└── requirements.txt
```
