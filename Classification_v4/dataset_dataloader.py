import pandas as pd
import h5py
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.transforms import v2

# Paths
# TRAIN_HDF5_PATH = "/kaggle/input/isic-2024-challenge/train-image.hdf5"
# TEST_HDF5_PATH = "/kaggle/input/isic-2024-challenge/test-image.hdf5"
# ANNOTATIONS_FILE = "/kaggle/input/isic-2024-challenge/train-metadata.csv"

TRAIN_HDF5_PATH = "E:\\isic-2024-challenge\\Dataset\\train-image.hdf5"
TEST_HDF5_PATH = "E:\\isic-2024-challenge\\Dataset\\test-image.hdf5"
ANNOTATIONS_FILE = "E:\\isic-2024-challenge\\Dataset\\train-metadata.csv"


# Image transformations and augmentations
TRAIN_TRANS = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.RandomRotation(degrees=(0, 360)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale = True)
])
TEST_TRANS = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True)
])


class ISIC2024_HDF5(Dataset):
    def __init__(self, hdf5_path, annotations_file=None, transform=None):
        self.hdf5_path = hdf5_path
        self.annotations_file = annotations_file
        self.transform = transform
        self.image_ids = []
        
        # Open HDF5 file once and keep it open
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        self.image_ids = list(self.hdf5_file.keys())

        if self.annotations_file is not None:
            self.labels = pd.read_csv(annotations_file, low_memory=False).set_index('isic_id')['target'].to_dict()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(BytesIO(self.hdf5_file[image_id][()]))

        if self.transform:
            image = self.transform(image)

        if self.annotations_file is not None:
            label = self.labels[image_id]
            return image, label, image_id
        else:
            return image, image_id

    def close(self):
        # Close the HDF5 file when done
        self.hdf5_file.close()


def get_loader(dataset_cls=ISIC2024_HDF5,
               train_hdf5_path=TRAIN_HDF5_PATH, 
               test_hdf5_path=TEST_HDF5_PATH, 
               train_labels_file=ANNOTATIONS_FILE, 
               train_img_trans=TRAIN_TRANS, 
               test_img_trans=TEST_TRANS, 
               batch=32, 
               seed=None):
    
    train_dataset_all = dataset_cls(hdf5_path=train_hdf5_path, annotations_file=train_labels_file, transform=train_img_trans)
    test_dataset = dataset_cls(hdf5_path=test_hdf5_path, transform=test_img_trans)

    train_annotations_all = pd.read_csv(train_labels_file, low_memory=False)
    labels = train_annotations_all['target']
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(train_annotations_all, labels))
    train_subset = Subset(train_dataset_all, train_idx)
    val_subset = Subset(train_dataset_all, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    
    return train_loader, val_loader, test_loader