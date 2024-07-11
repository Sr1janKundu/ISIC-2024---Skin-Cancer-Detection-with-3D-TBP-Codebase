'''
Importing modules
'''
import utils
import dataset_dataloader

'''
Constants
'''
IDX = 4     # get this index form train.lesgoo() or the row index with hightst avg_val_pAUC in log_folds.csv

'''
Paths
'''
# Kaggle
# TEST_HDF5_PATH = "/kaggle/input/isic-2024-challenge/test-image.hdf5"
# MODEL_SAVE_PATH = f"/kaggle/working/model_resnet34_aug_fold_{IDX}.pth"
# SUBMISSION_FILE_PATH = "/kaggle/working/submission.csv"

# Srijan
TEST_HDF5_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Data\\test-image.hdf5"
MODEL_SAVE_PATH = f"D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\model_resnet34_aug_fold_{IDX}.pth"
SUBMISSION_FILE_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Codebase\\Classification_v2\\submission.csv"

# Sruba
# TEST_HDF5_PATH = "E:\\isic-2024-challenge\\Dataset\\test-image.hdf5"
# MODEL_SAVE_PATH = f"E:\\isic-2024-challenge\\model_resnet34_aug_fold_{IDX}.pth"
# SUBMISSION_FILE_PATH = "E:\\isic-2024-challenge\\Codebase\\Classification_v2\\submission.csv"



def predict():
    model = utils.load_model(model_save_path=MODEL_SAVE_PATH)
    test_loader = dataset_dataloader.get_loader(test_hdf5_path=TEST_HDF5_PATH)
    utils.create_submission(model, test_loader, submission_file_path=SUBMISSION_FILE_PATH)


if __name__ == '__main__':
    predict()