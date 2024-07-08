import torch
import pandas as pd
from tqdm import tqdm
from dataset_dataloader import get_loader
import torchvision

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# MODEL_SAVE_PATH = "/kaggle/working/model_resnet34_aug_2-1.pth"
MODEL_SAVE_PATH = "E:\\isic-2024-challenge\\model_resnet34_aug_2-1.pth"
#BATCH_SIZE = 32
# SUBMISSION_FILE_PATH = "/kaggle/working/submission.csv"
SUBMISSION_FILE_PATH = "E:\\isic-2024-challenge\\Codebase\\Classification_v4\\submission.csv"


def load_model():
    model = torchvision.models.resnet34(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=1)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH)["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def create_submission(model, test_loader):
    predictions = []
    image_ids = []

    with torch.no_grad():
        for inputs, image_names in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())
            image_ids.extend(image_names)  # Append all image names from the batch

    # Check if the lengths match
    if len(image_ids) != len(predictions):
        print(f"Warning: Number of image IDs ({len(image_ids)}) does not match number of predictions ({len(predictions)})")

    # Create DataFrame
    submission_df = pd.DataFrame({
        'isic_id': image_ids,
        'target': predictions
    })

    # Save to CSV
    submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
    print(f"Submission file saved to {SUBMISSION_FILE_PATH}")

def main():
    print("Loading model...")
    model = load_model()
    
    print("Preparing test data...")
    _, _, test_loader = get_loader()
    
    print("Creating submission file...")
    create_submission(model, test_loader)

if __name__ == "__main__":
    main()