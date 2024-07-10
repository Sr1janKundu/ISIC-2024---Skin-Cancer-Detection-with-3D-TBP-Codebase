'''
Imports
'''
import h5py
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
# import cv2
from sklearn.model_selection import StratifiedShuffleSplit


'''
Transformations
'''
'''Transformations using torchvision.transforms.v2'''
data_transforms_v2 = {
    "train": v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.RandomRotation(degrees=(0, 360)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale = True)
    ]),
    "test": v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True)
    ])
}


data_transforms_album = {
    "train": A.Compose([
        A.Resize(224, 224),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=60, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "test": A.Compose([
        A.Resize(224, 224),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}


'''
DataClass
'''
class ISIC2024_HDF5(Dataset):
    '''
    with augmentations using torchvision.transforms.v2
    '''
    def __init__(self, hdf5_path, annotations_df=None, transform=None):
        self.hdf5_path = hdf5_path
        self.annotations_df = annotations_df
        self.transform = transform
        self.image_ids = []
        
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')

        if self.annotations_df is not None:
            self.image_ids = annotations_df['isic_id']
            self.labels = annotations_df.set_index('isic_id')['target'].to_dict()
        else:
            self.image_ids = list(self.hdf5_file.keys())
            

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]      # PIL
        image = Image.open(BytesIO(self.hdf5_file[image_id][()]))
        # image = self.load_image(self.hdf5_file[image_id][()])     # cv2

        if self.transform:
            image = self.transform(image)

        # Check for NaN in image
        if torch.isnan(image).any():
            print(f"NaN detected in image {image_id}")

        if self.annotations_df is not None:
            label = self.labels[image_id]
            # Check for NaN in label
            if np.isnan(label):
                print(f"NaN detected in label for image {image_id}")
            return image, label, image_id
        else:
            return image, image_id
        
    # def load_image(self, image_data):                             # cv2
    #     # Decode the image data from HDF5 file using OpenCV
    #     image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    #     # image = np.transpose(image, (1, 2, 0))  # Convert HxWxC to CxHxW
    #     return image
    
    def close(self):
        self.hdf5_file.close()


class ISIC2024_HDF5_ALBUM(Dataset):
    '''
    With augmentations using albumentations 
    '''
    def __init__(self, hdf5_path, annotations_df=None, transform=None):
        self.hdf5_path = hdf5_path
        self.annotations_df = annotations_df
        self.transform = transform
        self.image_ids = []
        
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')

        if self.annotations_df is not None:
            self.image_ids = annotations_df['isic_id']
            self.labels = annotations_df.set_index('isic_id')['target'].to_dict()
        else:
            self.image_ids = list(self.hdf5_file.keys())
            

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = np.array(Image.open(BytesIO(self.hdf5_file[image_id][()])))
        
        # image = self.load_image(self.hdf5_file[image_id][()])

        if self.transform:
            # image = self.transform(image)
            image = self.transform(image=image)["image"]        # Albumentations returns a dictionary with keys like 'image', 'mask', etc., depending on the transformations applied.

        # Check for NaN in image
        if torch.isnan(image).any():
            print(f"NaN detected in image {image_id}")

        if self.annotations_df is not None:
            label = self.labels[image_id]
            # Check for NaN in label
            if np.isnan(label):
                print(f"NaN detected in label for image {image_id}")
            return image, label, image_id
        else:
            return image, image_id
        
    # def load_image(self, image_data):
    #     # Decode the image data from HDF5 file using OpenCV
    #     image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    #     # image = np.transpose(image, (1, 2, 0))  # Convert HxWxC to CxHxW
    #     return image
    
    def close(self):
        self.hdf5_file.close()


'''
DataLoader
'''
def get_loader(test_hdf5_path, 
               train_labels_df = None, 
               train_hdf5_path = None, 
               dataset_cls=ISIC2024_HDF5_ALBUM,
               train_img_trans=data_transforms_album["train"], 
               test_img_trans=data_transforms_album["test"], 
               batch=32, 
               seed=None):
    if train_labels_df is not None and train_hdf5_path is not None:
        train_dataset_all = dataset_cls(hdf5_path=train_hdf5_path, annotations_df=train_labels_df, transform=train_img_trans)
        test_dataset = dataset_cls(hdf5_path=test_hdf5_path, transform=test_img_trans)

        train_annotations_all = train_labels_df
        labels = train_annotations_all['target']
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(splitter.split(train_annotations_all, labels))
        train_subset = Subset(train_dataset_all, train_idx)
        val_subset = Subset(train_dataset_all, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=False)

        return train_loader, val_loader, test_loader
    else:
        test_dataset = dataset_cls(hdf5_path=test_hdf5_path, transform=test_img_trans)
        test_loader = DataLoader(test_dataset, shuffle=False)
    
        return test_loader