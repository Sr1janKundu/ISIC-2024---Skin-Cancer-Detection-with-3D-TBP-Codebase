import h5py
import os
import shutil
from io import BytesIO
from PIL import Image
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torchvision.transforms import v2



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
    criterion = FocalLoss(alpha=0.25, gamma=2)  # Use the same FocalLoss as in training
    metric = BinaryF1Score(threshold=0.5).to(DEVICE)
    prec = BinaryPrecision(threshold=0.5).to(DEVICE)
    recall = BinaryRecall(threshold=0.5).to(DEVICE)
    acc = BinaryAccuracy(threshold=0.5).to(DEVICE)
    loss = 0.0
    num_corr = 0
    num_samp = 0
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels, _ in tqdm(loader):
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
    accu = float(num_corr) / float(num_samp)
    pauc = pauc_above_tpr(all_labels, all_preds)
    
    print(f"Total loss: {loss}, Average loss: {avg_loss}")
    print(f"Got {num_corr}/{num_samp} correct with accuracy {accu*100:.2f}")
    print(f"pAUC above 80% TPR: {pauc:.3f}, Accuracy: {acc.compute().item():.3f}, precision: {prec.compute().item():.3f}, recall: {recall.compute().item():.3f}, F1Score: {metric.compute().item():.3f}")
    model.train()

    return avg_loss, acc.compute().item(), prec.compute().item(), recall.compute().item(), metric.compute().item(), pauc

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def save_hdf5_as_jpg(hdf5_path, output_jpg_folder_path):
    if os.path.exists(output_jpg_folder_path):
        shutil.rmtree(output_jpg_folder_path)
    os.makedirs(output_jpg_folder_path)
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target):
        # Assuming output is of shape (batch_size,) and target is of shape (batch_size,)
        ce_loss = F.binary_cross_entropy_with_logits(output, target.float(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
        
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    fig.suptitle("Sample Images")
    
    for i in range(num_images):
        sample = dataset[i]
        if len(sample) == 3:  # (image, label, image_id)
            image, label, image_id = sample
        else:  # (image, image_id)
            image, image_id = sample
            label = "TEST"
        
        image = v2.ToPILImage()(image)  # Convert tensor back to PIL image if needed
        axes[i].imshow(image)
        axes[i].set_title(f"ISIC_ID: {image_id}\nTarget: {label}")
        axes[i].axis('off')
    plt.show()