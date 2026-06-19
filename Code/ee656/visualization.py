import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_sample(model, test_loader, device, num_samples=1):
    model.eval()
    with torch.no_grad():
        for i, (frames, labels) in enumerate(test_loader):
            if i >= num_samples:
                break
            frames, labels = frames.to(device), labels.to(device)

            # Forward pass
            logits, spatial_attn, temporal_attn = model(frames)

            # Handle spatial_attn and temporal_attn as lists
            # Average across layers
            spatial_attn = [attn.cpu().numpy() for attn in spatial_attn]  # List of (batch, heads, seq_len, h, w)
            spatial_attn = np.mean(spatial_attn, axis=0)  # Average layers: (batch, heads, seq_len, h, w)
            temporal_attn = [attn.cpu().numpy() for attn in temporal_attn]  # List of (batch, heads, seq_len, seq_len)
            temporal_attn = np.mean(temporal_attn, axis=0)  # Average layers: (batch, heads, seq_len, seq_len)

            # Visualize for first sample in batch
            for b in range(min(frames.size(0), num_samples)):
                # Spatial attention heatmap (average across heads)
                spatial_map = np.mean(spatial_attn[b], axis=0)  # Shape: (seq_len, h, w)
                for t in range(spatial_map.shape[0]):
                    plt.figure(figsize=(6, 6))
                    plt.imshow(spatial_map[t], cmap='hot', interpolation='nearest')
                    plt.title(
                        f'Spatial Attention Heatmap (Frame {t + 1}, Label: {"Real" if labels[b].item() == 0 else "Fake"})')
                    plt.colorbar()
                    plt.savefig(f'spatial_heatmap_sample_{i}_frame_{t}.png')
                    plt.close()

                # Temporal attention heatmap (average across heads)
                temporal_map = np.mean(temporal_attn[b], axis=0)  # Shape: (seq_len, seq_len)
                plt.figure(figsize=(6, 6))
                plt.imshow(temporal_map, cmap='hot', interpolation='nearest')
                plt.title(
                    f'Temporal Attention Heatmap (Sample {i + 1}, Label: {"Real" if labels[b].item() == 0 else "Fake"})')
                plt.colorbar()
                plt.savefig(f'temporal_heatmap_sample_{i}.png')
                plt.close()

                # Visualize input frame (first frame of sequence)
                frame = frames[b, 0].cpu().permute(1, 2, 0).numpy()  # (h, w, c)
                frame = (frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
                plt.figure(figsize=(6, 6))
                plt.imshow(frame)
                plt.title(f'Input Frame (Sample {i + 1}, Label: {"Real" if labels[b].item() == 0 else "Fake"})')
                plt.savefig(f'input_frame_sample_{i}.png')
                plt.close()


if __name__ == '__main__':
    from model import ISTVT
    from data_preprocessing_video import get_data_loaders
    import torch

    # Configuration
    data_dir = '/Users/sidharth/.cache/kagglehub/datasets/adityakeshri9234/uadfv-dataset/versions/1/UADFV'
    seq_len = 8  # Updated
    img_size = 224
    batch_size = 8
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load data and model
    train_loader, test_loader = get_data_loaders(data_dir, seq_len, img_size, batch_size)
    model = ISTVT(img_size=img_size, seq_len=seq_len).to(device)

    # Load trained weights (if available)
    try:
        model.load_state_dict(torch.load('trained_istvt.pth', map_location=device), strict=False)
    except FileNotFoundError:
        print("No trained weights found, using random weights")

    # Visualize
    visualize_sample(model, test_loader, device, num_samples=2)