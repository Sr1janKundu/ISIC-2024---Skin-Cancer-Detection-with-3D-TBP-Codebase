import csv
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from tqdm.auto import tqdm
from dataset_dataloader import get_loader
from utils import load_checkpoint, save_checkpoint, evaluate

# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
METADATA_CSV_PATH = "D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/Data/train-metadata.csv"
METADATA_CSV_PATH_50kSAMPLE = 'D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/EDA/isic2024_50ksample.csv'
MODEL_SAVE_PATH = "D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/model_resnet34_aug_2-1.pth"
LEARNING_RATE = 0.01
CLASSES = 1
EPOCH = 5

meta_df_sample = pd.read_csv(METADATA_CSV_PATH_50kSAMPLE, low_memory=False)

train_dl, val_dl, test_dl = get_loader(label_df=meta_df_sample, batch_size=BATCH_SIZE, seed=42)

model_resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(in_features=num_ftrs, out_features=1)
model_resnet.to(DEVICE)

def train(epochs, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    with open('log_res34_aug.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy', 'Validation Precision', 'Validation Recall', 'Validation F1 Score', 'Validation pAUC'])
        for epoch in range(epochs):
            print(f"\n | Epoch: {epoch+1}")
            total_loss = 0
            num_corr = 0
            num_samp = 0
            loop = tqdm(train_dl)
            for _, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze(1)
                preds = torch.sigmoid(outputs)
                num_corr += ((preds > 0.5) == labels).sum()
                num_samp += preds.size(0)
                loss = loss_func(outputs, labels.float())
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
            avg_loss = total_loss/len(train_dl)
            acc = num_corr / num_samp
            print(f"| Epoch {epoch+1}/{epochs} total training loss: {total_loss}, average training loss: {avg_loss}.")
            print("On Validation Data:")
            model.eval()
            with torch.inference_mode():
                val_loss, val_acc, val_pre, val_rec, val_f1, val_pauc = evaluate(val_dl, model)
            row = [epoch+1, avg_loss, acc.item(), val_loss, val_acc, val_pre, val_rec, val_f1, val_pauc]
            csv_writer.writerow(row)
            print('Saving model...')
            checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    }
            save_checkpoint(checkpoint, MODEL_SAVE_PATH)
            print(f'Model saved at {MODEL_SAVE_PATH}')
            model.train()

if __name__ == "__main__":
    # train(epochs=EPOCH, model=model_resnet)
    print("\n\nOn Test Data:")
    load_checkpoint(torch.load(MODEL_SAVE_PATH), model_resnet)
    model_resnet.eval()
    evaluate(test_dl, model_resnet)