import utils
import dataset_dataloader

IDX = 0

TEST_HDF5_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Data\\test-image.hdf5"
MODEL_SAVE_PATH = f"D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\model_vit_aug_fold_{IDX}.pth"
SUBMISSION_FILE_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Codebase\\Classification_v3\\submission.csv"

def predict():
    model = utils.load_model(model_save_path=MODEL_SAVE_PATH)
    test_loader = dataset_dataloader.get_loader(test_hdf5_path=TEST_HDF5_PATH)
    utils.create_submission(model, test_loader, submission_file_path=SUBMISSION_FILE_PATH)

if __name__ == '__main__':
    predict()