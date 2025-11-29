"""
Jaron Harbison
CS5680 – Final Project Update

This submission includes major updates to the original facial emotion
classification pipeline, implementing two new research components as
proposed in the project plan:

1. High-Resolution Region-Focused Image Construction
   ------------------------------------------------
   I replaced the original low-resolution holistic facial inputs with a
   new preprocessing method that extracts high-resolution facial regions
   (left eye, right eye, and mouth) directly from the input image.
   The system computes fixed region of interest coordinates based on
   normalized facial proportions and some testing based on input data.
   The three cropped regions are resized and combined into a vertically
   stacked “Frankenstein” image that preserves local detail while
   reducing overall input size. This improves the discriminatory power
   of the extracted features and allows the model to handle
   higher-resolution datasets.

   Modifications:
   - Increased resolution of the cropped regions of interest to preserve
     fine-grained texture (wrinkles, eyelid shape, mouth curvature).
   - Made the stacking format vertical to maintain more consistent feature
     alignment and improve CNN receptive field usage.
   - Added visualization utilities to display sample stacked images for
     debugging and qualitative evaluation.

2. Redesigned Convolutional Neural Network with Adaptive Pooling
   --------------------------------------------------------------
   The model architecture has been redesigned to support variable input
   resolutions using adaptive average pooling. The new network includes
   deep residual blocks, additional convolutional stages, and an
   AdaptiveAvgPool2d layer that guarantees a fixed-dimensional feature
   map (8x8), regardless of the input image size. This prevents
   dimension-mismatch errors and enables experimentation with multiple
   resolutions. The paper had a pre defined CNN model with dimensions
   but to fit the new higher resolution making it adaptive seemed best.

   Modifications:
   - Included additional convolutional layers to handle the higher-
     resolution input structure.
   - Replaced manually calculated flatten dimensions with adaptive
     pooling to eliminate shape errors and make the architecture
     resolution-agnostic.
   - Updated the fully connected layers accordingly.

These two research components represent major conceptual changes to the
original project and form the basis of the model evaluation and
experiments in the final report.
"""


import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt


# VERTICAL EYES + MOUTH CROPPING
def crop_eyes_mouth_vertical(img):
    h, w, _ = img.shape

    # Eye region
    eye_top = int(0.40 * h)
    eye_bottom = int(0.55 * h)
    left = int(0.30 * w)
    right = int(0.70 * w)

    left_eye = img[eye_top:eye_bottom, left:int((left+right)/2)]
    right_eye = img[eye_top:eye_bottom, int((left+right)/2):right]

    # Mouth region
    mouth_top = int(0.65 * h)
    mouth_bottom = int(0.85 * h)
    mouth = img[mouth_top:mouth_bottom, left:right]

    # Resize each to same width
    left_eye = cv2.resize(left_eye, (200, 80))
    right_eye = cv2.resize(right_eye, (200, 80))
    mouth = cv2.resize(mouth, (200, 80))

    # Grayscale
    left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
    right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
    mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)

    # Stack vertically: left eye, right eye, mouth
    combined = np.vstack([left_eye, right_eye, mouth])  # shape: (240, 200)
    return combined

# JAFFE dataset cropping eyes and mouth
class EmotionDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [f for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff'))]
        self.transform = transform
        self.mapping = {"AN": 0, "DI": 1, "FE": 2, "HA": 3, "NE": 4, "SA": 5, "SU": 6}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.folder, fname)
        img = cv2.imread(img_path)
        crop = crop_eyes_mouth_vertical(img)
        pil_img = Image.fromarray(crop)
        if self.transform:
            pil_img = self.transform(pil_img)
        emo_code = fname.split('.')[1][:2]
        label = self.mapping[emo_code]
        return pil_img, label

# CK+ CSV → Eyes/Mouth Crop Dataset
class CKPlusEyesMouthDataset(Dataset):
    def __init__(self, csv_file, transform=None, usage='Training'):
        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["Usage"] == usage]

        # EXCLUDE contempt (label = 7)
        self.df = self.df[self.df["emotion"] != 7]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert pixels to image
        pixels = np.array([int(p) for p in row["pixels"].split()], dtype=np.uint8)
        img = pixels.reshape(48, 48)

        # Upscale to mimic JAFFE format
        img = cv2.resize(img, (200, 200))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        crop = crop_eyes_mouth_vertical(img)

        pil_img = Image.fromarray(crop)
        if self.transform:
            pil_img = self.transform(pil_img)

        return pil_img, int(row["emotion"])

# TRANSFORMS (HIGHER RESOLUTION)
transform = T.Compose([
    T.Resize((256, 200)),
    T.ToTensor(),
])

# CNN MODEL (adaptive pooling for any input size since we are using high res images compared to the original size from the paper)
class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + x)
        return out


class FacialEmotionCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super().__init__()

        # --- Conv layers ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # downsample
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResidualBlock(64)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.res2 = ResidualBlock(128)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Adaptive pooling to fixed 8x8 feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x); x = self.pool1(x)
        x = self.conv2(x); x = self.res1(x)
        x = self.conv3(x); x = self.pool2(x); x = self.res2(x)
        x = self.conv4(x); x = self.pool3(x)
        x = self.conv5(x)
        x = self.adaptive_pool(x)   # <<< ensures fixed 8x8 feature map
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Show the stacked eyes+mouth image to see if the dimensions are good for images
# test_image = "jaffe/KA.AN1.39.tiff"
# img = cv2.imread(test_image)
# stacked = crop_eyes_mouth_vertical(img)
# plt.imshow(stacked, cmap='gray')
# plt.title("Stacked Eyes + Mouth")
# plt.axis('off')
# plt.show()

# training and evalution calls
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = FacialEmotionCNN().to(device)

    # JAFFE dataset
    jaffe_ds = EmotionDataset("jaffe/", transform=transform)

    # CK+ dataset
    ck_train_ds = CKPlusEyesMouthDataset("ckextended.csv", transform=transform, usage="Training")
    ck_test_ds = CKPlusEyesMouthDataset("ckextended.csv", transform=transform, usage="PublicTest")

    # Combine JAFFE & CK+
    combined_train = torch.utils.data.ConcatDataset([jaffe_ds, ck_train_ds])

    train_loader = DataLoader(combined_train, batch_size=16, shuffle=True)
    test_loader = DataLoader(ck_test_ds, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 25

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        train_loss = running_loss / total

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total * 100

        # report the epoch and training and testing
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")


    # run a little prediction for general testing
    test_image = "jaffe/KA.AN1.39.tiff"

    raw = cv2.imread(test_image)
    crop = crop_eyes_mouth_vertical(raw)
    pil_crop = Image.fromarray(crop)

    img_tensor = transform(pil_crop).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(img_tensor).argmax(dim=1).item()

    emotion_map = {0:"AN",1:"DI",2:"FE",3:"HA",4:"NE",5:"SA",6:"SU"}
    print("Predicted emotion:", emotion_map[pred])