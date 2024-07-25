import os
import csv
import pandas as pd
import utils
from dataset_dataloader import get_loader

TRAIN_HDF5_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Data\\train-image.hdf5"
TEST_HDF5_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Data\\test-image.hdf5"
ANNOTATIONS_FILE = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Data\\train-metadata.csv"
MODEL_SAVE_PATH_ = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\"
LOG_FILE_1 = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Codebase\\Classification_v3\\"
LOG_FILE_2 = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Codebase\\Classification_v3\\log_folds.csv"
SUBMISSION_FILE_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Codebase\\Classification_v3\\submission.csv"
METRICS_PLOT_SAVE_PATH = "D:\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\Codebase\\Classification_v3\\metrics.png"

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
CLASSES = 1
EPOCHS = 50
MIN_EPOCH_TRAIN = 10
PATIENCE = 5
EPSILON = 0.0005
NEG_POS_RATIO = 20
FOLDS = 1

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
            model_vit = utils.load_model()
            print(f"---------------\nTraining for fold: {fold+1}:\n---------------")
            val_pAUC_fold = utils.train(epochs=EPOCHS,
                                        model=model_vit,
                                        learning_rate=LEARNING_RATE,
                                        train_dl=train_dl,
                                        val_dl=val_dl,
                                        min_epoch_train=MIN_EPOCH_TRAIN,
                                        patience=PATIENCE,
                                        epsilon=EPSILON,
                                        log_file=os.path.join(LOG_FILE_1, f'log_vit_aug_fold_{fold}.csv'),
                                        model_save_path=os.path.join(MODEL_SAVE_PATH_, f'model_vit_aug_fold_{fold}.pth'))
            val_pAUC.append(val_pAUC_fold)
            csv_writer.writerow([fold, os.path.basename(os.path.join(MODEL_SAVE_PATH_, f'model_vit_aug_fold_{fold}.pth')), val_pAUC_fold])
    best_model_fold_index = val_pAUC.index(max(val_pAUC))
    
    file_paths = [os.path.join(LOG_FILE_1, f'log_vit_aug_fold_{i}.csv') for i in range(FOLDS)]
    utils.plot_metrics_from_files(file_paths, save_path=METRICS_PLOT_SAVE_PATH)
    
    return best_model_fold_index 

if __name__ == '__main__':
    indx = lesgooo()
    print(f"\n\nDone, index = {indx}\n")