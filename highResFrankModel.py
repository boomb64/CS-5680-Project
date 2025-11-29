import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt


# -------------------------------------------------------
#  VERTICAL EYES + MOUTH CROPPING (GRAYSCALE)
# -------------------------------------------------------
def crop_eyes_mouth_vertical(img):
    """Crop left eye → right eye → mouth, stack vertically, return grayscale image."""

    h, w = img.shape  # grayscale only

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

    # Resize all crops to identical width
    left_eye = cv2.resize(left_eye, (200, 80))
    right_eye = cv2.resize(right_eye, (200, 80))
    mouth = cv2.resize(mouth, (200, 80))

    # Stack vertically: L-eye → R-eye → mouth  → (240, 200)
    combined = np.vstack([left_eye, right_eye, mouth])

    return combined  # still grayscale


# -------------------------------------------------------
#  JAFFE DATASET (GRAYSCALE)
# -------------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [f for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff'))]
        self.transform = transform

        self.mapping = {
            "AN": 0, "DI": 1, "FE": 2, "HA": 3,
            "NE": 4, "SA": 5, "SU": 6
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.folder, fname)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        crop = crop_eyes_mouth_vertical(img)

        pil = Image.fromarray(crop)

        if self.transform:
            pil = self.transform(pil)

        # Extract emotion from filename
        parts = fname.split(".")
        emo_code = None
        for p in parts:
            if p[:2] in self.mapping:
                emo_code = p[:2]
                break

        label = self.mapping[emo_code]
        return pil, label


# -------------------------------------------------------
#  CK+ CSV DATASET (GRAYSCALE)
# -------------------------------------------------------
class CKPlusEyesMouthDataset(Dataset):
    def __init__(self, csv_file, transform=None, usage='Training'):
        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["Usage"] == usage]

        # REMOVE contempt (emotion 7)
        self.df = self.df[self.df["emotion"] != 7]

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert pixel string to 48x48 grayscale image
        pixels = np.array(row["pixels"].split(), dtype=np.uint8)
        img = pixels.reshape(48, 48)

        # Upscale to avoid tiny crops
        img = cv2.resize(img, (200, 200))

        crop = crop_eyes_mouth_vertical(img)

        pil = Image.fromarray(crop)

        if self.transform:
            pil = self.transform(pil)

        return pil, int(row["emotion"])


# -------------------------------------------------------
#  TRANSFORMS
# -------------------------------------------------------
transform = T.Compose([
    T.Resize((256, 200)),
    T.ToTensor(),  # now (1, H, W)
])


# -------------------------------------------------------
#  CNN ARCHITECTURE
# -------------------------------------------------------
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
        return self.relu(out + x)


class FacialEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
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
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.res2 = ResidualBlock(128)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.adapt = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()

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
        x = self.adapt(x)
        x = self.flatten(x)
        return self.fc(x)


# -------------------------------------------------------
#  OPTIONAL: Show a sample Frankenstein image
# -------------------------------------------------------
def show_sample():
    img_path = "jaffe/KA.AN1.39.tiff"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    stacked = crop_eyes_mouth_vertical(img)
    plt.imshow(stacked, cmap='gray')
    plt.title("Stacked Eyes + Eyes + Mouth")
    plt.axis("off")
    plt.show()


# -------------------------------------------------------
#  TRAINING ONLY — NO TESTING
# -------------------------------------------------------
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    model = FacialEmotionCNN().to(device)

    # datasets
    jaffe = EmotionDataset("jaffe/", transform=transform)
    ck_train = CKPlusEyesMouthDataset("ckextended.csv", transform=transform, usage="Training")

    # combined training set
    train_ds = ConcatDataset([jaffe, ck_train])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # loss + optimizer
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

        train_loss = running_loss / total
        train_acc = correct / total * 100

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

    # save trained weights
    torch.save(model.state_dict(), "highres_frankenstein.pth")
    print("Model saved as highres_frankenstein.pth")
