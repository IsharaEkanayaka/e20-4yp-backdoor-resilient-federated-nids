import torch
from torch.utils.data import TensorDataset, DataLoader
import os

# Update this path to match your structure
PROCESSED_DATA_PATH = "data/unsw-nb15/processed/train_pool.pt"
TEST_DATA_PATH = "data/unsw-nb15/processed/global_test.pt"

def load_dataset(path=PROCESSED_DATA_PATH):
    """
    Loads a saved .pt file and returns a TensorDataset.
    Returns: (dataset, input_dim, num_classes)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Data file not found at {path}. Run src.data.preprocess first!")

    print(f"📂 Loading data from {path}...")
    data = torch.load(path, weights_only=True)
    
    X = data['X']
    y = data['y']
    
    # Calculate dimensions automatically
    input_dim = X.shape[1]      # Should be 72
    num_classes = len(torch.unique(y)) # Should be 10
    
    dataset = TensorDataset(X, y)
    
    return dataset, input_dim, num_classes

def get_data_loaders(batch_size=32):
    """
    Helper to get simple loaders for the Server's IID testing.
    """
    # Load Global Test Set
    test_ds, _, _ = load_dataset(TEST_DATA_PATH)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Load Train Pool (Architect will split this later, but useful for debugging)
    train_ds, input_dim, num_classes = load_dataset(PROCESSED_DATA_PATH)
    
    return train_ds, test_loader, input_dim, num_classes