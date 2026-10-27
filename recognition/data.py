from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def image_transform(image_size: int = 128, train: bool = False, rotation_degrees: float = 0.0):
    operations = [transforms.Resize((image_size, image_size))]
    if train and rotation_degrees > 0:
        operations.append(transforms.RandomRotation(rotation_degrees, fill=0))
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)


class IdentityFolderDataset(Dataset):
    def __init__(self, root: str | Path, transform=None, max_identities: int = 0):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Training root does not exist: {self.root}")
        identities = sorted(directory.name for directory in self.root.iterdir() if directory.is_dir())
        if max_identities:
            identities = identities[:max_identities]
        self.class_to_idx = {identity: index for index, identity in enumerate(identities)}
        self.samples = []
        for identity in identities:
            for path in sorted((self.root / identity).rglob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[identity]))
        if not self.samples:
            raise ValueError(f"No images found under {self.root}")
        self.transform = transform or image_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            # Both grayscale CASIA ROIs and gray-semantic RGB PCE images enter ResNet as RGB.
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, str(path)


class PairImageDataset(Dataset):
    def __init__(self, pairs_csv: str | Path, transform=None):
        self.pairs_csv = Path(pairs_csv).resolve()
        with self.pairs_csv.open(newline="", encoding="utf-8") as handle:
            self.records = list(csv.DictReader(handle))
        self.paths = sorted({record[key] for record in self.records for key in ("path1", "path2")})
        self.transform = transform or image_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path
