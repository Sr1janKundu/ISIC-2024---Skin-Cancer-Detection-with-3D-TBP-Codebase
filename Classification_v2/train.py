'''
Imports
'''
import os
import csv
import pandas as pd

'''
Importing modules
'''
import utils
from dataset_dataloader import get_loader


# Paths
## Kaggle
# TRAIN_HDF5_PATH = "/kaggle/input/isic-2024-challenge/train-image.hdf5"
# TEST_HDF5_PATH = "/kaggle/input/isic-2024-challenge/test-image.hdf5"
# ANNOTATIONS_FILE = "/kaggle/input/isic-2024-challenge/train-metadata.csv"
# MODEL_SAVE_PATH_ = "/kaggle/working/"
# LOG_FILE_1 = "/kaggle/working/"
# LOG_FILE_2 = "/kaggle/working/log_folds.csv"
# RESNET34_IMAGENET_WEIGHTS_PYTORCH = "/kaggle/input/resnet34-weights/pytorch/nan/1/resnet34-b627a593.pth"        # change properly
# SUBMISSION_FILE_PATH = "/kaggle/working/submission.csv"
# METRICS_PLOT_SAVE_PATH = "/kaggle/working/metrics.png"

## Local_Srijan
TRAIN_HDF5_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Data\\train-image.hdf5"
TEST_HDF5_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Data\\test-image.hdf5"
ANNOTATIONS_FILE = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Data\\train-metadata.csv"
MODEL_SAVE_PATH_ = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\"
LOG_FILE_1 = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Codebase\\Classification_v2\\"
LOG_FILE_2 = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Codebase\\Classification_v2\\log_folds.csv"
RESNET34_IMAGENET_WEIGHTS_PYTORCH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\resnet34-b627a593.pth"        # change properly
SUBMISSION_FILE_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Codebase\\Classification_v2\\submission.csv"
METRICS_PLOT_SAVE_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Codebase\\Classification_v2\\metrics.png"

## Local_Sruba
# TRAIN_HDF5_PATH = "E:\\isic-2024-challenge\\Dataset\\train-image.hdf5"
# TEST_HDF5_PATH = "E:\\isic-2024-challenge\\Dataset\\test-image.hdf5"
# ANNOTATIONS_FILE = "E:\\isic-2024-challenge\\Dataset\\train-metadata.csv"
# MODEL_SAVE_PATH_ = "E:\\isic-2024-challenge\\"
# LOG_FILE_1 = "E:\\isic-2024-challenge\\Codebase\\Classification_v2\\"
# LOG_FILE_2 = "E:\\isic-2024-challenge\\Codebase\\Classification_v2\\log_folds.csv"
# RESNET34_IMAGENET_WEIGHTS_PYTORCH = "E:\\isic-2024-challenge\\resnet34-b627a593.pth"        # change properly
# SUBMISSION_FILE_PATH = "E:\\isic-2024-challenge\\Codebase\\Classification_v2\\submission.csv"
# METRICS_PLOT_SAVE_PATH = "E:\\isic-2024-challenge\\Codebase\\Classification_v2\\metrics.png"


'''
Constants
'''
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
CLASSES = 1
EPOCHS = 50
MIN_EPOCH_TRAIN = 10
PATIENCE = 5
EPSILON = 0.0025
NEG_POS_RATIO = 20
FOLDS = 3


def lesgooo():
    annotations_df_full = pd.read_csv(ANNOTATIONS_FILE, low_memory=False)
    df_positive_all = annotations_df_full[annotations_df_full["target"] == 1].reset_index(drop=True)
    df_negative_all = annotations_df_full[annotations_df_full["target"] == 0].reset_index(drop=True)
    df_negative_trunc = df_negative_all.sample(df_positive_all.shape[0]*NEG_POS_RATIO)
    annotations_df_trunc = pd.concat([df_positive_all, df_negative_trunc]).sample(frac=1).reset_index()
    val_pAUC = []
    with open(LOG_FILE_2, 'w', newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Fold', "Model_Name", "avg_val_pAUC"])
        for fold in range(FOLDS):
            train_dl, val_dl, _ = get_loader(train_labels_df=annotations_df_trunc,
                                             train_hdf5_path=TRAIN_HDF5_PATH,
                                             test_hdf5_path=TEST_HDF5_PATH)
            model_resnet = utils.load_model(imagenet_weights_path=RESNET34_IMAGENET_WEIGHTS_PYTORCH)
            print(f"---------------\nTraining for fold: {fold+1}:\n---------------")
            val_pAUC_fold = utils.train(epochs=EPOCHS,
                                        model=model_resnet,
                                        learning_rate=LEARNING_RATE,
                                        train_dl=train_dl,
                                        val_dl=val_dl,
                                        min_epoch_train=MIN_EPOCH_TRAIN,
                                        patience=PATIENCE,
                                        epsilon=EPSILON,
                                        log_file=os.path.join(LOG_FILE_1, f'log_res34_aug_fold_{fold}.csv'),
                                        model_save_path=os.path.join(MODEL_SAVE_PATH_, f'model_resnet34_aug_fold_{fold}.pth'))
            val_pAUC.append(val_pAUC_fold)
            csv_writer.writerow([fold, os.path.basename(os.path.join(MODEL_SAVE_PATH_, f'model_resnet34_aug_fold_{fold}.pth')), val_pAUC_fold]) # Logging avg pauc for the model trained on each fold
    best_model_fold_index = val_pAUC.index(max(val_pAUC))
    
    file_paths = [os.path.join(LOG_FILE_1, f'log_res34_aug_fold_{i}.csv') for i in range(5)]
    utils.plot_metrics_from_files(file_paths, save_path=METRICS_PLOT_SAVE_PATH)
    
    return best_model_fold_index 

if __name__ == '__main__':
    indx = lesgooo()
    print(f"\n\nDone, index = {indx}\n")