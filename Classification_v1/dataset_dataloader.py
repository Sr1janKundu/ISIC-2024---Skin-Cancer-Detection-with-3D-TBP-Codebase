import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch, torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
from PIL import Image


# Paths
IMAGE_DIR = "D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/Data/train-image/image"  # Directory with images
ANNOTATIONS_FILE = "D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/EDA/isic2024_50ksample.csv"
# ANNOTATIONS_FILE = "D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/Data/train-metadata.csv"  # CSV file with image names and labels

TRANS = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.RandomRotation(degrees=(0, 360)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale = True)
])


class ISIC2024(Dataset):
    def __init__(self, image_dir, annotations_file, extension='.jpg', transform=None):
        # Specify the data types for the relevant columns
        dtype = {'isic_id': str, 'target': int}
        self.image_dir = image_dir
        self.annotations = pd.read_csv(annotations_file, dtype=dtype, low_memory=False)
        self.extension = extension
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.annotations.iloc[idx, 0] + self.extension)
        image = Image.open(img_name).convert("RGB")
        label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label
    

def get_loader(dataset = ISIC2024, data_dir = IMAGE_DIR, labels_file = ANNOTATIONS_FILE, img_trans = TRANS, batch = 32, seed = None):
    isic2024_dataset = dataset(image_dir=data_dir, annotations_file=labels_file, transform=img_trans)
    annotations = pd.read_csv(labels_file, low_memory=False)
    labels = annotations['target']
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(annotations, labels))
    train_subset = Subset(isic2024_dataset, train_idx)
    test_subset = Subset(isic2024_dataset, test_idx)
    train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch, shuffle=False)

    return train_loader, test_loader

