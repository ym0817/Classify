import os,torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, root_dir, type_names,transform=None):
        """
        root_dir: Directory with 'images' folder containing class subfolders (e.g., 'crazing', 'inclusion')
        transform: PyTorch transforms for preprocessing and augmentation
        """
        # self.root_dir = Path(root_dir) / 'images'
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = type_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Images directory {self.root_dir} does not exist.")
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.is_dir():
                print(f"Warning: Directory {cls_dir} not found. Skipping.")
                continue
            for img_name in cls_dir.glob('*.jpg'):
                if img_name.is_file():
                    self.images.append(str(img_name))
                    self.labels.append(self.class_to_idx[cls])
                else:
                    print(f"Warning: {img_name} is not a valid file. Skipping.")
        if not self.images:
            raise ValueError(f"No valid images found in {self.root_dir}.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path)#.convert('L')  # Grayscale
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, 256, 256)), label
