import torch
from model import ISTVT
from data_preprocessing_video import get_data_loaders
from train import train_model
from visualization import visualize_sample


def main():
    # Configuration
    data_dir = '/Users/sidharth/.cache/kagglehub/datasets/adityakeshri9234/uadfv-dataset/versions/1/UADFV'
    seq_len = 8
    img_size = 224
    batch_size = 8
    num_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    train_loader, test_loader = get_data_loaders(data_dir, seq_len, img_size, batch_size)
    if train_loader is None or test_loader is None:
        print("Failed to load dataset")
        return

    # Model initialization
    model = ISTVT(img_size=img_size, seq_len=seq_len)

    # Train model
    trained_model = train_model(model, train_loader, test_loader, num_epochs, device)

    # Visualize heatmaps for a sample
    visualize_sample(trained_model, test_loader, device)


if __name__ == '__main__':
    main()