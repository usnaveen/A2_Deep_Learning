"""Dataset loader for Oxford-IIIT Pet — multi-task (classification, bbox, trimap).

Handles:
  - Auto-download via torchvision
  - Breed class labels (37 classes)
  - Head bounding boxes  (x_center, y_center, w, h  in pixel space)
  - Trimap segmentation masks (3 classes: foreground=1, background=2, boundary=3 → remapped to 0,1,2)
  - Normalization (ImageNet stats — required by the autograder)
  - Augmentations via albumentations

The dataset is structured to support all three tasks simultaneously.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# ImageNet normalization (autograder expects normalized inputs)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────────────
# Helper: download and organise the raw Oxford-IIIT Pet data
# ──────────────────────────────────────────────────────────────────────────────

def download_oxford_pet(root: str = "./data/oxford_pet"):
    """Download Oxford-IIIT Pet images, annotations, and trimaps if not present."""
    import tarfile
    import urllib.request

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    images_dir = root / "images"
    annots_dir = root / "annotations"

    if images_dir.exists() and annots_dir.exists():
        return str(root)

    base_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
    files = ["images.tar.gz", "annotations.tar.gz"]

    for fname in files:
        url = base_url + fname
        dst = root / fname
        if not dst.exists():
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, str(dst))
        # Extract
        print(f"Extracting {fname} ...")
        with tarfile.open(str(dst)) as tar:
            tar.extractall(str(root))

    return str(root)


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224):
    """Training augmentations (image + mask + bboxes)."""
    if not HAS_ALBUMENTATIONS:
        return None
    return A.Compose(
        [
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(max_holes=4, max_height=image_size // 8, max_width=image_size // 8,
                            min_holes=1, fill_value=0, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"], min_visibility=0.3),
        additional_targets={"mask": "mask"},
    )


def get_val_transforms(image_size: int = 224):
    """Validation / test transforms (deterministic)."""
    if not HAS_ALBUMENTATIONS:
        return None
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"], min_visibility=0.3),
        additional_targets={"mask": "mask"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Each sample returns:
        image:  [3, H, W] normalised tensor
        label:  int in [0, 36]  (breed class)
        bbox:   [4]  tensor  (x_center, y_center, w, h) in pixel space after resize
        mask:   [H, W] long tensor with values in {0, 1, 2}
    """

    CLASS_NAMES = [
        "Abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound",
        "beagle", "Bengal", "Birman", "Bombay", "boxer", "British_Shorthair",
        "chihuahua", "Egyptian_Mau", "english_cocker_spaniel", "english_setter",
        "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin",
        "keeshond", "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland",
        "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue",
        "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu",
        "Siamese", "Sphynx", "staffordshire_bull_terrier", "wheaten_terrier",
        "yorkshire_terrier",
    ]

    def __init__(
        self,
        root: str = "./data/oxford_pet",
        split: str = "trainval",
        image_size: int = 224,
        transform=None,
        download: bool = True,
    ):
        """
        Args:
            root: Root directory for the raw dataset.
            split: 'trainval' or 'test' (matches the official list.txt files).
            image_size: Resize dimension.
            transform: Albumentations Compose (if None, uses default train/val).
            download: Whether to download the dataset if not present.
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        if download:
            download_oxford_pet(root)

        self.images_dir = self.root / "images"
        self.annots_dir = self.root / "annotations"
        self.trimaps_dir = self.annots_dir / "trimaps"
        self.xmls_dir = self.annots_dir / "xmls"

        # Parse split file
        list_file = self.annots_dir / f"{split}.txt"
        self.samples = self._parse_list(list_file)

        # Build class name → index mapping from file names
        self._build_class_map()

        # Transforms
        if transform is not None:
            self.transform = transform
        elif split == "trainval":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

    def _parse_list(self, list_file: Path):
        """Parse the official train/test split list."""
        samples = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                name = parts[0]       # e.g. Abyssinian_1
                class_id = int(parts[1]) - 1   # 1-indexed → 0-indexed
                samples.append((name, class_id))
        return samples

    def _build_class_map(self):
        """Build breed name → class idx from parsed samples."""
        # Class name is everything except the last _N
        self.name_to_idx = {}
        for name, class_id in self.samples:
            breed = "_".join(name.split("_")[:-1])
            self.name_to_idx[breed] = class_id

    def _load_bbox(self, name: str, orig_w: int, orig_h: int) -> np.ndarray:
        """Load bounding box from XML annotation.

        Returns: [xmin, ymin, xmax, ymax] in original image pixel coords.
        """
        xml_path = self.xmls_dir / f"{name}.xml"
        if xml_path.exists():
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            obj = root.find("object")
            if obj is not None:
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                # Clip to image bounds
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(orig_w, xmax)
                ymax = min(orig_h, ymax)
                return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

        # Fallback: full image as bbox
        return np.array([0, 0, orig_w, orig_h], dtype=np.float32)

    def _load_trimap(self, name: str) -> np.ndarray:
        """Load trimap segmentation mask.

        Original trimap values: 1=foreground, 2=background, 3=boundary
        Remapped to: 0=foreground, 1=background, 2=boundary
        """
        trimap_path = self.trimaps_dir / f"{name}.png"
        if trimap_path.exists():
            mask = np.array(Image.open(trimap_path))
            # Remap 1→0, 2→1, 3→2
            mask = mask - 1
            mask = np.clip(mask, 0, 2)
            return mask.astype(np.uint8)

        # Fallback: all background
        return np.ones((self.image_size, self.image_size), dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name, class_id = self.samples[idx]

        # Load image
        img_path = self.images_dir / f"{name}.jpg"
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image_np = np.array(image)

        # Load annotations
        bbox_xyxy = self._load_bbox(name, orig_w, orig_h)  # [xmin, ymin, xmax, ymax]
        mask = self._load_trimap(name)

        # Resize mask to match image if needed
        if mask.shape[:2] != image_np.shape[:2]:
            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
            mask = np.array(mask_pil)

        # Apply albumentations transform
        if self.transform is not None:
            transformed = self.transform(
                image=image_np,
                mask=mask,
                bboxes=[bbox_xyxy.tolist()],
                bbox_labels=[class_id],
            )
            image_tensor = transformed["image"]               # [3, H, W]
            _mask = transformed["mask"]
            if torch.is_tensor(_mask):
                mask_tensor = _mask.long()
            else:
                mask_tensor = torch.from_numpy(_mask).long()  # [H, W]

            if len(transformed["bboxes"]) > 0:
                bbox_resized = list(transformed["bboxes"][0])  # [xmin, ymin, xmax, ymax] in resized space
            else:
                # Bbox was lost during transform (e.g. flipped out) — fallback
                bbox_resized = [0.0, 0.0, float(self.image_size), float(self.image_size)]
        else:
            # Manual fallback (no albumentations)
            from torchvision import transforms as T
            tf = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            image_tensor = tf(image)
            mask_pil = Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_pil)).long()

            # Scale bbox
            sx = self.image_size / orig_w
            sy = self.image_size / orig_h
            bbox_resized = [
                bbox_xyxy[0] * sx, bbox_xyxy[1] * sy,
                bbox_xyxy[2] * sx, bbox_xyxy[3] * sy,
            ]

        # Convert bbox from (xmin, ymin, xmax, ymax) → (xcenter, ycenter, w, h)
        xmin, ymin, xmax, ymax = bbox_resized
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        bbox_tensor = torch.tensor([cx, cy, w, h], dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": torch.tensor(class_id, dtype=torch.long),
            "bbox": bbox_tensor,
            "mask": mask_tensor,
            "name": name,
        }


def create_dataloaders(
    root: str = "./data/oxford_pet",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.15,
    seed: int = 42,
    download: bool = True,
):
    """Create train, validation, and test dataloaders.

    The official 'trainval' split is further split into train/val.
    The 'test' split is used as-is.
    """
    from torch.utils.data import DataLoader, random_split

    # Full trainval dataset
    trainval_dataset = OxfordIIITPetDataset(
        root=root, split="trainval", image_size=image_size, download=download
    )

    # Split trainval into train + val
    n = len(trainval_dataset)
    n_val = int(n * val_split)
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(trainval_dataset, [n_train, n_val], generator=generator)

    # Override transform for val subset to be deterministic
    # (random_split returns Subset, so we need to handle transforms carefully)
    # We'll use the training transforms for train and val transforms for val
    # by creating separate dataset instances
    train_ds = OxfordIIITPetDataset(
        root=root, split="trainval", image_size=image_size,
        transform=get_train_transforms(image_size), download=False,
    )
    val_ds = OxfordIIITPetDataset(
        root=root, split="trainval", image_size=image_size,
        transform=get_val_transforms(image_size), download=False,
    )

    # Use the same indices from the split
    from torch.utils.data import Subset
    train_dataset = Subset(train_ds, train_dataset.indices)
    val_dataset = Subset(val_ds, val_dataset.indices)

    # Test dataset
    test_dataset = OxfordIIITPetDataset(
        root=root, split="test", image_size=image_size,
        transform=get_val_transforms(image_size), download=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader