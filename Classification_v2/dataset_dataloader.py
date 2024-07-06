import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
from torchvision.io import decode_jpeg
from sklearn.model_selection import train_test_split
import pandas as pd
import io
from PIL import Image
import numpy as np

TRAIN_DATA_PATH = 'D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/Data/train-image.hdf5'
TEST_DATA_PATH = 'D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/Data/test-image.hdf5'
METADATA_CSV_PATH = "D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/Data/train-metadata.csv"

TRANS_TRAIN = v2.Compose([
    v2.ToImage(),
    v2.RandomRotation(degrees=(0, 360)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale = True)
])

TRANS_TEST = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale = True)
])

def preprocess_labels(label_df):
    label_df['isic_id'] = label_df['isic_id'].str.replace('.jpg', '')
    return dict(zip(label_df['isic_id'], label_df['target']))

class ISIC_2024_HDF5Dataset(Dataset):
    def __init__(self, file_path, labels_dict=None, transform=None, cache_size=1000):
        self.file_path = file_path
        self.transform = transform
        self.file = h5py.File(self.file_path, 'r')
        self.keys = list(self.file.keys())
        self.labels_dict = labels_dict
        self.cache = {}
        self.cache_size = cache_size
        
        if labels_dict is not None:
            self.keys = [key for key in self.keys if key in labels_dict]
    
    def __getitem__(self, index):
        key = self.keys[index]
        try:
            if key in self.cache:
                image = self.cache[key]
            else:
                jpeg_data = self.file[key][()]
                image = Image.open(io.BytesIO(jpeg_data))
                image = np.array(image)
                if len(self.cache) < self.cache_size:
                    self.cache[key] = image

            if self.transform:
                image = self.transform(image)

            image = v2.Resize((224, 224))(image)

            if self.labels_dict:
                label = self.labels_dict[key]
                return image, torch.tensor(label, dtype=torch.float32)
            else:
                return image, torch.tensor(0, dtype=torch.float32)  # For test set without labels
        except Exception as e:
            print(f"Error loading image {key}: {str(e)}")
            return self.__getitem__((index + 1) % len(self))  # Try next image
    
    def __len__(self):
        return len(self.keys)
    
    def __del__(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()

def get_loader(label_df, batch_size=16, seed=None):
    labels_dict = preprocess_labels(label_df) if label_df is not None else None
    train_dataset = ISIC_2024_HDF5Dataset(TRAIN_DATA_PATH, labels_dict=labels_dict, transform=TRANS_TRAIN)
    test_dataset = ISIC_2024_HDF5Dataset(TEST_DATA_PATH, transform=TRANS_TEST)
    
    train_idx, val_idx = train_test_split(
        range(len(train_dataset)),
        test_size=0.2,
        stratify=[train_dataset[i][1].item() for i in range(len(train_dataset))],
        random_state=seed
    )
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# class ISIC_2024_HDF5Dataset(Dataset):
#     def __init__(self, file_path, label_csv=None, transform=None):
#         self.file_path = file_path
#         self.transform = transform
        
#         with h5py.File(self.file_path, 'r') as file:
#             self.keys = list(file.keys())
        
#         if label_csv:
#             self.labels_df = pd.read_csv(label_csv, low_memory=False)
#             self.labels_df['isic_id'] = self.labels_df['isic_id'].str.replace('.jpg', '')
#             self.labels_dict = dict(zip(self.labels_df['isic_id'], self.labels_df['target']))
#         else:
#             self.labels_dict = None
    
#     def __getitem__(self, index):
#         with h5py.File(self.file_path, 'r') as file:
#             key = self.keys[index]
#             jpeg_data = file[key][()]
#             image = Image.open(io.BytesIO(jpeg_data))
#             image = np.array(image)
        
#         if self.transform:
#             image = self.transform(image)
        
#         if self.labels_dict:
#             label = self.labels_dict.get(key, -1)  # Use -1 if key not found
#             return image, label
#         else:
#             return image, 0  # For test set without labels
    
#     def __len__(self):
#         return len(self.keys)

# def get_loader(batch_size=16, seed=None):
#     train_dataset = ISIC_2024_HDF5Dataset(TRAIN_DATA_PATH, label_csv=METADATA_CSV_PATH, transform=TRANS_TRAIN)
#     test_dataset = ISIC_2024_HDF5Dataset(TEST_DATA_PATH, transform=TRANS_TEST)
    
#     # Split train dataset into train and validation
#     train_idx, val_idx = train_test_split(
#         range(len(train_dataset)),
#         test_size=0.2,
#         stratify=[train_dataset[i][1] for i in range(len(train_dataset))],
#         random_state=seed
#     )
    
#     train_subset = Subset(train_dataset, train_idx)
#     val_subset = Subset(train_dataset, val_idx)
    
#     train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, test_loader