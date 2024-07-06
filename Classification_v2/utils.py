import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, auc

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = 1

def pauc_above_tpr(y_true, y_pred, min_tpr=0.80):
    y_true = abs(np.array(y_true) - 1)
    y_pred = -1.0 * np.array(y_pred)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    max_fpr = 1 - min_tpr

    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    
    if len(fpr) < 2:
        print("Warning: Not enough points to compute pAUC. Returning 0.")
        return 0
    
    partial_auc = auc(fpr, tpr)

    return partial_auc

def evaluate(loader, model):
    criterion = nn.BCEWithLogitsLoss()
    metric = BinaryF1Score(threshold=0.5).to(DEVICE)
    prec = BinaryPrecision(threshold=0.5).to(DEVICE)
    recall = BinaryRecall(threshold=0.5).to(DEVICE)
    acc = BinaryAccuracy(threshold=0.5).to(DEVICE)
    loss = 0.0
    num_corr = 0
    num_samp = 0
    all_preds = []
    all_labels = []
    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs).squeeze(1)
        preds = torch.sigmoid(outputs)
        num_corr += ((preds > 0.5) == labels).sum()
        num_samp += preds.size(0)
        loss += criterion(outputs, labels.float()).item()
        metric.update(preds, labels)
        prec.update(preds, labels)
        recall.update(preds, labels)
        acc.update(preds, labels)
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = loss / len(loader)
    accu = float(num_corr)/float(num_samp)
    pauc = pauc_above_tpr(all_labels, all_preds)
    
    print(f"Total loss: {loss}, Average loss: {avg_loss}")
    print(f"Got {num_corr}/{num_samp} correct with accuracy {accu*100:.2f}")
    print(f"pAUC above 80% TPR: {pauc:.3f}, Accuracy: {acc.compute():.3f}, precision: {prec.compute():.3f}, recall: {recall.compute():.3f}, F1Score: {metric.compute():.3f}")
    model.train()

    return avg_loss, acc.compute().item(), prec.compute().item(), recall.compute().item(), metric.compute().item(), pauc

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])