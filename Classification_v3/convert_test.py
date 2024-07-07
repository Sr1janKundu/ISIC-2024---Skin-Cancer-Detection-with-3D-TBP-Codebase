import h5py
import os
from io import BytesIO
from PIL import Image


def save_hdf5_as_jpg(hdf5_path, output_jpg_folder_path):
    with h5py.File(hdf5_path, 'r') as hdf5_f:
        for image_name in hdf5_f:
            img = hdf5_f[image_name]
            image_bytes = img[()]
            try:
                image = Image.open(BytesIO(image_bytes))
                output_filename = os.path.join(output_jpg_folder_path, f"{image_name}.jpg")
                image.save(output_filename, 'JPEG')
            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")