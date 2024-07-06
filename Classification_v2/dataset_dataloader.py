import h5py
import io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from PIL import Image

TRAIN_DATA_PATH = 'D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/Data/train-image.hdf5'
TEST_DATA_PATH = 'D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/Data/test-image.hdf5'
# METADATA_CSV_PATH_50kSAMPLE = 'D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/EDA/isic2024_50ksample.csv'
METADATA_CSV_PATH = "D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/Data/train-metadata.csv"

TRANS_TRAIN = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.RandomRotation(degrees=(0, 360)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True)
])

TRANS_TEST = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])


# class ISIC_2024_HDF5Dataset(Dataset):
#     def __init__(self, file_path, label_df=None, transform=None):
#         self.file_path = file_path
#         self.transform = transform
        
#         with h5py.File(self.file_path, 'r') as file:
#             all_keys = list(file.keys())
        
#         if label_df is not None:
#             # Ensure 'isic_id' doesn't have the .jpg extension
#             label_df['isic_id'] = label_df['isic_id'].str.replace('.jpg', '')
            
#             # Filter keys to only those present in the label_df
#             self.keys = [key for key in all_keys if key in label_df['isic_id'].values]
            
#             # Create a dictionary for quick label lookup
#             self.labels_dict = dict(zip(label_df['isic_id'], label_df['target']))
#         else:
#             self.keys = all_keys
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
#             label = self.labels_dict[key]
#             return image, label
#         else:
#             return image, 0  # For test set without labels
    
#     def __len__(self):
#         return len(self.keys)


# def get_loader(label_df, batch_size=16, seed=None):
#     train_dataset = ISIC_2024_HDF5Dataset(TRAIN_DATA_PATH, label_df=label_df, transform=TRANS_TRAIN)
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


class ISIC_2024_HDF5Dataset(Dataset):
    def __init__(self, file_path, label_csv=None, transform=None):
        self.file_path = file_path
        self.transform = transform
        
        with h5py.File(self.file_path, 'r') as file:
            self.keys = list(file.keys())
        
        if label_csv:
            self.labels_df = pd.read_csv(label_csv, low_memory=False)
            self.labels_df['isic_id'] = self.labels_df['isic_id'].str.replace('.jpg', '')
            self.labels_dict = dict(zip(self.labels_df['isic_id'], self.labels_df['target']))
        else:
            self.labels_dict = None
    
    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as file:
            key = self.keys[index]
            jpeg_data = file[key][()]
            image = Image.open(io.BytesIO(jpeg_data))
            image = np.array(image)
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels_dict:
            label = self.labels_dict.get(key, -1)  # Use -1 if key not found
            return image, label
        else:
            return image, 0  # For test set without labels
    
    def __len__(self):
        return len(self.keys)

def get_loader(batch_size=16, seed=None):
    train_dataset = ISIC_2024_HDF5Dataset(TRAIN_DATA_PATH, label_csv=METADATA_CSV_PATH, transform=TRANS_TRAIN)
    test_dataset = ISIC_2024_HDF5Dataset(TEST_DATA_PATH, transform=TRANS_TEST)
    
    # Split train dataset into train and validation
    train_idx, val_idx = train_test_split(
        range(len(train_dataset)),
        test_size=0.2,
        stratify=[train_dataset[i][1] for i in range(len(train_dataset))],
        random_state=seed
    )
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader