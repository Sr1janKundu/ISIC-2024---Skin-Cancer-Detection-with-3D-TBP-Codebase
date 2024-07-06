import torch
import pandas as pd
import h5py
from tqdm import tqdm
from dataset_dataloader import ISIC_2024_HDF5Dataset, TRANS_TEST, TEST_DATA_PATH
from torch.utils.data import DataLoader
import torchvision

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_PATH = "D:/ISIC 2024 - Skin Cancer Detection with 3D-TBP/model_resnet34_aug_2-1.pth"
BATCH_SIZE = 32
SUBMISSION_FILE_PATH = "submission.csv"

def load_model():
    model = torchvision.models.resnet34(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=1)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH)["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

def get_test_loader():
    test_dataset = ISIC_2024_HDF5Dataset(TEST_DATA_PATH, transform=TRANS_TEST)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader

def create_submission(model, test_loader):
    predictions = []
    image_ids = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())

    # Get image IDs from the HDF5 file
    with h5py.File(TEST_DATA_PATH, 'r') as file:
        image_ids = list(file.keys())

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
    test_loader = get_test_loader()
    
    print("Creating submission file...")
    create_submission(model, test_loader)

if __name__ == "__main__":
    main()