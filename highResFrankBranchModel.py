import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# ========================================================================
#  HAAR CASCADE–BASED FACE PART CROPPING MODULE
# ------------------------------------------------------------------------
# This module implements the key preprocessing step of the “Frankenstein”
# high-resolution input method. It extracts the left eye, right eye, and
# mouth regions from each image. These cropped regions are stacked to
# create a single composite 128×128×3 patch used as input to the CNN.
#
# Improvements included:
#   • Added Haar cascades for more stable feature localization
#   • Added fallback geometric cropping when detection fails
#   • Re-tuned cropping ratios for higher-resolution images
# ========================================================================

# Load pretrained Haar cascades (OpenCV built-in resources)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def crop_face_parts(image, out_size=(128, 128)):
    """
    Extracts the left eye, right eye, and mouth regions using a combination
    of Haar cascades and geometric heuristics, then returns a vertically
    stacked composite crop suitable for input into the neural network.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or color face image.
    out_size : tuple(int,int)
        Target size (height, width) to resize each facial region to.

    Returns
    -------
    stacked : np.ndarray
        Composite array containing [left eye; right eye; mouth] stacked
        vertically. Shape = (3*out_size[0], out_size[1]).
    """

    # Convert to grayscale if needed
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # -------------------------------
    # Face detection using Haar model
    # -------------------------------
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Use first detected face
        x, y, fw, fh = faces[0]
        face_roi = gray[y:y+fh, x:x+fw]
    else:
        # Fallback: center-crop default region
        # This ensures the dataset never returns None.
        fw, fh = int(w*0.5), int(h*0.5)
        x, y = (w-fw)//2, (h-fh)//2
        face_roi = gray[y:y+fh, x:x+fw]

    # ===============================================================
    # EYE REGION CROPPING (Geometric)
    # These ratios were manually tuned for better alignment when using
    # higher-resolution images, addressing the paper's limitations.
    # ===============================================================

    eye_top = int(fh * 0.25)          # shifted downward to avoid eyebrow area
    eye_height = int(fh * 0.22)
    le_x, le_w = int(fw*0.15), int(fw*0.25)
    re_x, re_w = int(fw*0.60), int(fw*0.25)

    left_eye  = face_roi[eye_top:eye_top+eye_height, le_x:le_x+le_w]
    right_eye = face_roi[eye_top:eye_top+eye_height, re_x:re_x+re_w]

    # ===============================================================
    # MOUTH REGION CROPPING
    # Adjusted for stability across mixed datasets (JAFFE + CK+).
    # ===============================================================
    mouth_y_start = int(fh * 0.65)
    mouth_y_end = min(int(fh * 0.92), fh)
    mouth_roi = face_roi[mouth_y_start:mouth_y_end, :]

    # fallback for extremely small faces
    if mouth_roi.size == 0:
        mouth_roi = face_roi[fh//2 : fh, :]

    # Resize crops
    left_eye  = cv2.resize(left_eye, out_size)
    right_eye = cv2.resize(right_eye, out_size)
    mouth_roi = cv2.resize(mouth_roi, out_size)

    # Stack vertically into Frankenstein input
    stacked = np.vstack([left_eye, right_eye, mouth_roi])
    return stacked

# ========================================================================
#  JAFFE DATASET LOADER
# ------------------------------------------------------------------------
# Loads images directly from the JAFFE folder. Each image is converted to
# a Frankenstein composite using crop_face_parts(). The dataset uses a
# retry loop so that failed detections never interrupt training.
# ========================================================================

class EmotionDataset(Dataset):
    """
    PyTorch dataset for loading emotion-labeled images from JAFFE.
    """

    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [f for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff'))]
        self.transform = transform

        # Mapping from JAFFE filename codes to emotion ID
        self.mapping = {"AN":0,"DI":1,"FE":2,"HA":3,"NE":4,"SA":5,"SU":6}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns transformed Frankenstein image and emotion label.
        Includes a retry loop so cropping never returns None.
        """

        while True:
            fname = self.files[idx]
            img_path = os.path.join(self.folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            crop = crop_face_parts(img)
            if crop is not None:
                break

            # Try new sample if detection fails
            idx = np.random.randint(0, len(self.files))

        pil = Image.fromarray(crop)
        if self.transform:
            pil = self.transform(pil)

        # Extract emotion code from filename
        emo_code = next((p[:2] for p in fname.split(".")
                         if p[:2] in self.mapping), None)

        return pil, self.mapping[emo_code]

# ========================================================================
#  CK+ CSV DATASET LOADER
# ------------------------------------------------------------------------
# Loads CK+ facial expression images stored as flattened CSV pixel data.
# Resizes images, generates Frankenstein crops, and returns tensors.
# ========================================================================

class CKPlusEyesMouthDataset(Dataset):
    """
    Loads CK+ images from a CSV file in the FER2013-like format.
    Only labels 0-6 are retained (emotion 7 = contempt removed).
    """

    def __init__(self, csv_file, transform=None, usage='Training'):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["Usage"] == usage]
        self.df = self.df[self.df["emotion"] != 7]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Loads pixel string, reshapes to 48×48, resizes, and returns
        the Frankenstein feature crop plus emotion label.
        Includes retry loop to ensure stable cropping.
        """
        while True:
            row = self.df.iloc[idx]
            pixels = np.array(row["pixels"].split(), dtype=np.uint8)
            img = pixels.reshape(48, 48)
            img = cv2.resize(img, (200, 200))

            crop = crop_face_parts(img)
            if crop is not None:
                break

            idx = np.random.randint(0, len(self.df))

        pil = Image.fromarray(crop)
        if self.transform:
            pil = self.transform(pil)

        return pil, int(row["emotion"])

# ========================================================================
#  IMAGE TRANSFORM PIPELINE
# ========================================================================

transform = T.Compose([
    T.Resize((256, 200)),   # Final resize applied to Frankenstein input
    T.ToTensor(),           # Convert to PyTorch tensor
])

# ========================================================================
#  CNN ARCHITECTURE
# ------------------------------------------------------------------------
# Residual CNN inspired by the paper’s Extended DNN architecture.
# Improvements include:
#   • Higher capacity (32→512 channels)
#   • Residual blocks for deeper representation
#   • AdaptiveAvgPool for variable input compatibility
#   • Dropout regularization for generalization
# ========================================================================

class ResidualBlock(nn.Module):
    """
    Basic residual block: Conv → BN → ReLU → Conv → BN → Add → ReLU.
    """

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
    """
    Convolutional neural network designed to classify Frankenstein
    feature-stacked inputs into 7 facial emotion categories.
    """

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

        self.adapt = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()

        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(512*8*8, 1024),
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

# ========================================================================
#  TRAINING + VALIDATION LOOP
# ------------------------------------------------------------------------
# Combines JAFFE + CK+ datasets, splits into train/test, and runs the full
# 35-epoch training process while reporting loss and accuracy per epoch.
# ========================================================================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    model = FacialEmotionCNN().to(device)

    # JAFFE dataset (80/20 split)
    jaffe_full = EmotionDataset("jaffe/", transform=transform)
    train_size = int(0.8 * len(jaffe_full))
    test_size = len(jaffe_full) - train_size
    jaffe_train, jaffe_test = random_split(jaffe_full, [train_size, test_size])

    # CK+ dataset (explicit Usage field)
    ck_train = CKPlusEyesMouthDataset("ckextended.csv", transform=transform, usage="Training")
    ck_test  = CKPlusEyesMouthDataset("ckextended.csv", transform=transform, usage="PublicTest")

    # Merge datasets for mixed training
    train_ds = ConcatDataset([jaffe_train, ck_train])
    test_ds  = ConcatDataset([jaffe_test, ck_test])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    EPOCHS = 35

    # -----------------------------
    # Main training loop
    # -----------------------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        # Training step
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total * 100

        # -----------------------------
        # Validation step
        # -----------------------------
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total * 100

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

    # Save model weights
    torch.save(model.state_dict(), "highres_frankenstein.pth")
    print("Model saved as highres_frankenstein.pth")
