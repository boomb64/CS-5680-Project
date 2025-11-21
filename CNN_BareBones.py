# since we just talked with you this afternoon about how to change the project
# we haven't yet had a chance to start on the seperate code
# here is the CNN and model that we worked together to make, next week our
# code will be very different, but we didn't have time to
# write completely different code in one day
# this code will be both of our starting points

import os
import pandas as pd
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    # starts immplementing the CNN as the paper has it

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        # First unit
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(in_channels)

        # Second unit
        self.conv2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(in_channels)
        self.conv2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # First unit
        shortcut1 = x
        out = self.relu(self.bn1_1(self.conv1_1(x)))
        out = self.bn1_2(self.conv1_2(out))
        out += shortcut1
        out = self.relu(out)

        # Second unit
        shortcut2 = out
        out = self.relu(self.bn2_1(self.conv2_1(out)))
        out = self.bn2_2(self.conv2_2(out))
        out += shortcut2
        out = self.relu(out)

        return out


class FacialEmotionCNN(nn.Module):
    # CNN from the paper using pytorch
    def __init__(self, in_channels=1, num_classes=6):
        super(FacialEmotionCNN, self).__init__()

        # Input: (1, 128, 96)
        self.conv1 = nn.Sequential(
            # 5x5 filter, stride 2. Padding=2 to get 64x48 output
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # Out: (32, 64, 48)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Out: (32, 32, 24)

        self.conv2 = nn.Sequential(
            # 3x3 filter, stride 1. Padding=1 to keep size
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # Out: (64, 32, 24)

        self.res1 = ResidualBlock(in_channels=64)  # Out: (64, 32, 24)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # Out: (128, 32, 24)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Out: (128, 16, 12)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # Out: (128, 16, 12)

        self.res2 = ResidualBlock(in_channels=128)  # Out: (128, 16, 12)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # Out: (256, 16, 12)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Out: (256, 8, 6)

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # Out: (512, 8, 6)

        self.flatten = nn.Flatten()  # Out: 512 * 8 * 6 = 24576

        # Calculate flattened size
        self.fc_input_features = 512 * 8 * 6

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Paper mentions dropout
        )  # Out: 1024

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Paper mentions dropout
        )  # Out: 512

        self.output_layer = nn.Linear(512, num_classes)  # Out: 6

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.conv6(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output_layer(x)

        return x


emotion_map = {
    "AN": 0,  # Angry
    "DI": 1,  # Disgust
    "FE": 2,  # Fear
    "HA": 3,  # Happy
    "NE": 4,  # Neutral
    "SA": 5,  # Sad
    "SU": 6  # Surprise
}

transform = T.Compose([
    T.ToTensor(),
])


class JAFFEDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        self.files = [f for f in os.listdir(folder_path) if f.endswith(('.tiff', '.jpg', '.png'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.folder_path, filename)

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (96, 128))
        img = img[:, :, np.newaxis]

        # Extract emotion label
        emotion_code = filename.split(".")[1][:2]  # AN, FE, HA, etc.
        label = emotion_map[emotion_code]

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label).long()


class CKPlusDataset(Dataset):
    def __init__(self, csv_file, transform=None, usage='PublicTest'):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['Usage'] == usage]
        self.transform = transform

        # Filter out 'Contempt' if your model has 7 classes
        # (the CK+ dataset has 1 emotion that JAFFE doesn't)
        self.data = self.data[self.data['emotion'] != 7]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
        img = pixels.astype(np.uint8)
        img = cv2.resize(img, (96, 128))
        img = img[:, :, np.newaxis]

        if self.transform:
            img = self.transform(img)

        label = int(row['emotion'])
        return img, torch.tensor(label).long()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = FacialEmotionCNN(num_classes=7).to(device)

    # Transforms
    transform_train = T.Compose([
        T.ToTensor(),
    ])
    transform_ck = T.Compose([
        T.ToTensor(),
    ])

    # Load datasets
    jaffe_ds = JAFFEDataset("jaffe/", transform=transform_train)
    ck_train_ds = CKPlusDataset("ckextended.csv", transform=transform_ck, usage='Training')
    ck_test_ds = CKPlusDataset("ckextended.csv", transform=transform_ck, usage='PublicTest')

    # Combine JAFFE and CK+ training data
    combined_train_ds = torch.utils.data.ConcatDataset([jaffe_ds, ck_train_ds])

    # DataLoaders
    train_loader = DataLoader(combined_train_ds, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(ck_test_ds, batch_size=16, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    EPOCHS = 25

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = total_loss / total

        # Validation on CK+ PublicTest
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # these predicition will probably be changed in the future
    # especially as we diverge projects, this is just to make sure it works

    # example prediction against a single image
    test_image_path = "jaffe/KA.AN1.39.tiff"  # replace with any JAFFE image
    img = cv2.imread(test_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (96, 128))
    img = img[:, :, np.newaxis]
    img = transform_train(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(img)
        class_id = pred.argmax(dim=1).item()

    emotion_names = {v: k for k, v in emotion_map.items()}
    print("Predicted emotion:", emotion_names[class_id])

    # does a final test agagst the CK+ dataset
    ck_loader = DataLoader(ck_test_ds, batch_size=16, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in ck_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"CK+ Test Accuracy: {correct / total * 100:.2f}%")