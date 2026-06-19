import torch
import torch.nn as nn
import torch.optim as optim
from model import ISTVT
from data_preprocessing_video import get_data_loaders


def train_model(model, train_loader, test_loader, num_epochs=10, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _, _ = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            scheduler.step()

            running_loss += loss.item() * frames.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for frames, labels in test_loader:
                frames, labels = frames.to(device), labels.to(device)
                logits, _, _ = model(frames)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

    return model
# Add to train.py, after training loop
    torch.save(model.state_dict(), '/Users/sidharth/Desktop/SidPhantom/ee656/trained_istvt.pth')


if __name__ == '__main__':
    data_dir = '/Users/sidharth/.cache/kagglehub/datasets/adityakeshri9234/uadfv-dataset/versions/1/UADFV'
    seq_len = 8
    img_size = 224
    batch_size = 8
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_data_loaders(data_dir, seq_len, img_size, batch_size)
    if train_loader is None or test_loader is None:
        print("Failed to load dataset")


    else :
        model = ISTVT(img_size=img_size, seq_len=seq_len)
        trained_model = train_model(model, train_loader, test_loader, num_epochs, device)