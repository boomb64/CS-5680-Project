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

# -------------------------------------------------------
#  HAAR CASCADE–BASED FACE PART CROPPING
# -------------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def crop_face_parts(image, out_size=(128, 128)):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Face detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, fw, fh = faces[0]
        face_roi = gray[y:y+fh, x:x+fw]
    else:
        # fallback: center crop
        fw, fh = int(w*0.5), int(h*0.5)
        x, y = (w-fw)//2, (h-fh)//2
        face_roi = gray[y:y+fh, x:x+fw]

    # Adjusted geometric eye regions
    eye_top = int(fh * 0.25)
    eye_height = int(fh * 0.22)
    le_x, le_w = int(fw*0.15), int(fw*0.25)
    re_x, re_w = int(fw*0.6), int(fw*0.25)
    left_eye  = face_roi[eye_top:eye_top+eye_height, le_x:le_x+le_w]
    right_eye = face_roi[eye_top:eye_top+eye_height, re_x:re_x+re_w]

    # Geometric mouth region
    mouth_y_start = int(fh*0.65)
    mouth_y_end   = min(int(fh*0.92), fh)
    mouth_roi = face_roi[mouth_y_start:mouth_y_end, :]
    if mouth_roi.size == 0:
        mouth_roi = face_roi[fh//2:fh, :]

    # Resize all
    left_eye  = cv2.resize(left_eye, out_size)
    right_eye = cv2.resize(right_eye, out_size)
    mouth_roi = cv2.resize(mouth_roi, out_size)

    stacked = np.vstack([left_eye, right_eye, mouth_roi])
    return stacked

# -------------------------------------------------------
#  JAFFE DATASET
# -------------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff'))]
        self.transform = transform
        self.mapping = {"AN":0,"DI":1,"FE":2,"HA":3,"NE":4,"SA":5,"SU":6}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        while True:
            fname = self.files[idx]
            img_path = os.path.join(self.folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            crop = crop_face_parts(img)
            if crop is not None:
                break
            idx = np.random.randint(0, len(self.files))
        pil = Image.fromarray(crop)
        if self.transform:
            pil = self.transform(pil)
        emo_code = next((p[:2] for p in fname.split(".") if p[:2] in self.mapping), None)
        label = self.mapping[emo_code]
        return pil, label

# -------------------------------------------------------
#  CK+ CSV DATASET
# -------------------------------------------------------
class CKPlusEyesMouthDataset(Dataset):
    def __init__(self, csv_file, transform=None, usage='Training'):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["Usage"] == usage]
        self.df = self.df[self.df["emotion"] != 7]  # remove contempt
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
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

# -------------------------------------------------------
#  TRANSFORMS
# -------------------------------------------------------
transform = T.Compose([
    T.Resize((256, 200)),
    T.ToTensor(),
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
        self.conv1 = nn.Sequential(nn.Conv2d(1,32,5,stride=2,padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.res1 = ResidualBlock(64)
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2)
        self.res2 = ResidualBlock(128)
        self.conv4 = nn.Sequential(nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(2)
        self.conv5 = nn.Sequential(nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.adapt = nn.AdaptiveAvgPool2d((8,8))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512*8*8,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
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
#  TRAIN/VALIDATION SETUP
# -------------------------------------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    model = FacialEmotionCNN().to(device)

    # JAFFE split
    jaffe_full = EmotionDataset("jaffe/", transform=transform)
    train_size = int(0.8*len(jaffe_full))
    test_size = len(jaffe_full) - train_size
    jaffe_train, jaffe_test = random_split(jaffe_full, [train_size, test_size])

    # CK+ train/val based on Usage
    ck_train = CKPlusEyesMouthDataset("ckextended.csv", transform=transform, usage="Training")
    ck_test = CKPlusEyesMouthDataset("ckextended.csv", transform=transform, usage="PublicTest")

    # Combine datasets
    train_ds = ConcatDataset([jaffe_train, ck_train])
    test_ds = ConcatDataset([jaffe_test, ck_test])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    EPOCHS = 25

    # -------------------------------------------------------
    #  TRAINING LOOP WITH VALIDATION
    # -------------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0
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

        # Validation
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total * 100
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    torch.save(model.state_dict(), "highres_frankenstein.pth")
    print("Model saved as highres_frankenstein.pth")
