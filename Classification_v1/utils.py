import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, AUROC
import torch.nn as nn
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = 1


def evaluate(loader, model):
    criterion = nn.BCEWithLogitsLoss()
    metric = BinaryF1Score(threshold=0.5).cuda()
    prec = BinaryPrecision(threshold=0.5).cuda()
    recall = BinaryRecall(threshold=0.5).cuda()
    acc = BinaryAccuracy(threshold=0.5).cuda()
    auc = AUROC(task="binary").cuda()
    loss = 0.0
    num_corr = 0
    num_samp = 0
    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs).squeeze(1)
        preds = torch.round(outputs.sigmoid())
        num_corr += (preds == labels).sum()
        num_samp += preds.size(0)
        loss += criterion(outputs, labels.float()).item()
        metric.update(preds, labels)
        prec.update(preds, labels)
        recall.update(preds, labels)
        acc.update(preds, labels)
        auc.update(outputs, labels)
    avg_loss = loss / len(loader)
    accu = float(num_corr)/float(num_samp)
    print(f"Total loss: {loss}, Average loss: {avg_loss}")
    print(f"Got {num_corr}/{num_samp} correct with accuracy {accu*100:.2f}")
    print(f"| AUC: {auc.compute():.3f}, Accuracy: {acc.compute():.3f}, precision: {prec.compute():.3f}, recall: {recall.compute():.3f}, F1Score: {metric.compute():.3f}")
    model.train()

    return avg_loss, acc.compute().item(), prec.compute().item(), recall.compute().item(), metric.compute().item(), auc.compute().item()


def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])