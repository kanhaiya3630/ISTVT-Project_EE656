# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import os
# import cv2
# from PIL import Image
# import numpy as np
#
#
# class DeepfakeVideoDataset(Dataset):
#     def __init__(self, data_dir, seq_len=2, img_size=224, transform=None):
#         self.data_dir = data_dir
#         self.seq_len = seq_len
#         self.img_size = img_size
#         self.transform = transform
#         self.videos = []
#         self.labels = []
#
#         # Load videos from 'real' and 'fake' subfolders
#         for label in ['real', 'fake']:
#             label_dir = os.path.join(data_dir, label)
#             if not os.path.exists(label_dir):
#                 print(f"Warning: Directory {label_dir} does not exist")
#                 continue
#             for video in os.listdir(label_dir):
#                 if video.endswith('.mp4'):
#                     video_path = os.path.join(label_dir, video)
#                     self.videos.append(video_path)
#                     self.labels.append(0 if label == 'real' else 1)
#
#         if not self.videos:
#             raise ValueError(f"No .mp4 files found in {data_dir}")
#
#     def __len__(self):
#         return len(self.videos)
#
#     def __getitem__(self, idx):
#         video_path = self.videos[idx]
#         label = self.labels[idx]
#
#         # Load video and extract frames
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print(f"Error: Failed to open video {video_path}")
#             return torch.zeros(self.seq_len, 3, self.img_size, self.img_size), label
#
#         frames = []
#         frame_count = 0
#         while len(frames) < self.seq_len and cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if frame_count % 2 == 0:  # Sample every 5th frame
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#                 img = Image.fromarray(frame)
#                 if self.transform:
#                     img = self.transform(img)
#                 frames.append(img)
#             frame_count += 1
#         cap.release()
#
#         # Pad with last frame or zeros if insufficient frames
#         while len(frames) < self.seq_len:
#             frames.append(frames[-1] if frames else torch.zeros(3, self.img_size, self.img_size))
#
#         # Stack frames
#         frames = torch.stack(frames, dim=0)  # (seq_len, c, h, w)
#         return frames, label
#
#
# def get_data_loaders(data_dir, seq_len=2, img_size=224, batch_size=8):
#     transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     try:
#         train_dataset = DeepfakeVideoDataset(os.path.join(data_dir, 'train'), seq_len, img_size, transform)
#         test_dataset = DeepfakeVideoDataset(os.path.join(data_dir, 'test'), seq_len, img_size, transform)
#     except ValueError as e:
#         print(f"Error creating dataset: {e}")
#         return None, None
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, test_loader


import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class DeepfakeVideoDataset(Dataset):
    def __init__(self, data_dir, seq_len=8, img_size=224, transform=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.img_size = img_size
        self.transform = transform
        self.videos = []
        self.labels = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        for label in ['real', 'fake']:
            label_dir = os.path.join(data_dir, label)
            if not os.path.exists(label_dir):
                print(f"Warning: Directory {label_dir} does not exist")
                continue
            for video in os.listdir(label_dir):
                if video.endswith('.mp4'):
                    video_path = os.path.join(label_dir, video)
                    self.videos.append(video_path)
                    self.labels.append(0 if label == 'real' else 1)

        if not self.videos:
            raise ValueError(f"No .mp4 files found in {data_dir}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Failed to open video {video_path}")
            return torch.zeros(self.seq_len, 3, self.img_size, self.img_size), label

        frames = []
        frame_count = 0
        while len(frames) < self.seq_len and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 1 == 0:  # Sample every 2nd frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Detect face
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    frame = frame[y:y + h, x:x + w]
                img = Image.fromarray(frame)
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            frame_count += 1
        cap.release()

        while len(frames) < self.seq_len:
            frames.append(frames[-1] if frames else torch.zeros(3, self.img_size, self.img_size))

        frames = torch.stack(frames, dim=0)
        return frames, label


def get_data_loaders(data_dir, seq_len=8, img_size=224, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = DeepfakeVideoDataset(os.path.join(data_dir, 'train'), seq_len, img_size, transform)
        test_dataset = DeepfakeVideoDataset(os.path.join(data_dir, 'test'), seq_len, img_size, transform)
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        return None, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader