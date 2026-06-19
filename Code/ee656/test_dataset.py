import torch
from data_preprocessing_video import get_data_loaders

# Configuration
data_dir = '/Users/sidharth/.cache/kagglehub/datasets/adityakeshri9234/uadfv-dataset/versions/1/UADFV'
seq_len = 2
img_size = 224
batch_size = 8

# Get data loaders
train_loader, test_loader = get_data_loaders(data_dir, seq_len, img_size, batch_size)

# Test loading a batch
if train_loader is None or test_loader is None:
    print("Failed to create data loaders")
else:
    try:
        frames, labels = next(iter(train_loader))
        print(f"Successfully loaded batch: frames shape={frames.shape}, labels shape={labels.shape}")
    except Exception as e:
        print(f"Error loading batch: {e}")