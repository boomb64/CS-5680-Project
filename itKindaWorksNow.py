import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import cv2
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter


# ---------------------------------------------------------
# 1. Preprocessing Transforms
# ---------------------------------------------------------

class FaceDetector(object):
    """
    Finds the face in a large image and crops to it.
    Added to TRAINING to ensure consistency with Inference.
    """

    def __init__(self, min_size=60):
        try:
            # Requires opencv-python to be installed
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.active = True
        except:
            print("Warning: Frontal face cascade not found. Skipping face detection.")
            self.active = False
        self.min_size = min_size

    def __call__(self, img):
        if not self.active:
            return img

        # Safety Check: If image is already small (e.g. 48x48 pixel-based dataset),
        # skip detection to avoid errors or bad crops on already-cropped data.
        h, w = img.shape[:2]
        if h < self.min_size or w < self.min_size:
            return img

        # Detect faces
        faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return img  # Return original if no face found

        # Assume the largest face is the target
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        return img[y:y + h, x:x + w]


class PaperBasedCrop(object):
    # does the cropping based on paper geometry
    def __init__(self, output_size=(128, 96)):
        self.target_size = (96, 128)

    def __call__(self, img):
        h, w = img.shape[:2]

        # Simulated landmarks (assuming centered face), from paper
        left_eye_center = np.array([w * 0.35, h * 0.4])
        right_eye_center = np.array([w * 0.65, h * 0.4])

        eyes_mid_point = (left_eye_center + right_eye_center) / 2
        a = np.linalg.norm(eyes_mid_point - right_eye_center)

        # Paper dimensions logic
        top = int(eyes_mid_point[1] - (1.4 * a))
        bottom = int(eyes_mid_point[1] + (3.3 * a))
        left = int(eyes_mid_point[0] - (2.5 * a))
        right = int(eyes_mid_point[0] + (2.5 * a))

        # Padding
        top = max(0, top)
        left = max(0, left)
        bottom = min(h, bottom)
        right = min(w, right)

        # Numpy slicing [y:y+h, x:x+w]
        img_cropped = img[top:bottom, left:right]

        # Handle case where crop might be empty due to bad landmarks/image
        if img_cropped.size == 0:
            return cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        return cv2.resize(img_cropped, self.target_size, interpolation=cv2.INTER_LINEAR)


class SmartFeatureStitcher(object):
    """
    Attempts to detect Eyes and Mouth using Haar Cascades to create a precise crop.
    Falls back to fixed percentages if detection fails.
    """

    def __init__(self, output_size=(96, 128)):
        self.output_size = output_size
        try:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            self.active = True
        except:
            self.active = False

    def __call__(self, img):
        h, w = img.shape[:2]

        eyes_strip = None
        mouth_strip = None

        if self.active:
            # 1. ATTEMPT DETECTION
            # Detect Eyes (search in top 60% of image)
            top_half = img[:int(h * 0.6), :]
            eyes = self.eye_cascade.detectMultiScale(top_half, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

            # Detect Mouth (search in bottom 50% of image)
            bottom_half = img[int(h * 0.5):, :]
            mouths = self.mouth_cascade.detectMultiScale(bottom_half, scaleFactor=1.1, minNeighbors=5, minSize=(25, 15))

            # -- EYE LOGIC --
            if len(eyes) >= 1:
                min_y = np.min(eyes[:, 1])
                max_y_h = np.max(eyes[:, 1] + eyes[:, 3])
                pad = 10
                y1 = max(0, min_y - pad)
                y2 = min(int(h * 0.6), max_y_h + pad)
                eyes_strip = img[y1:y2, :]

            # -- MOUTH LOGIC --
            if len(mouths) >= 1:
                mouths = sorted(mouths, key=lambda x: x[2] * x[3], reverse=True)
                mx, my, mw, mh = mouths[0]
                y_offset = int(h * 0.5)
                real_y = my + y_offset
                pad = 10
                y1 = max(y_offset, real_y - pad)
                y2 = min(h, real_y + mh + pad)
                mouth_strip = img[y1:y2, :]

        # 2. FALLBACKS
        if eyes_strip is None:
            eyes_strip = img[int(h * 0.15):int(h * 0.50), :]
        if mouth_strip is None:
            mouth_strip = img[int(h * 0.65):int(h * 0.95), :]

        # 3. STITCH
        try:
            combined_img = np.vstack((eyes_strip, mouth_strip))
        except ValueError:
            return cv2.resize(img, self.output_size, interpolation=cv2.INTER_LINEAR)

        # 4. FINAL RESIZE
        return cv2.resize(combined_img, self.output_size, interpolation=cv2.INTER_LINEAR)


class ToTensorAndFixDims(object):
    # converts to a tensor
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            if len(pic.shape) == 2:
                pic = pic[:, :, None]
            return T.functional.to_tensor(pic)
        return T.functional.to_tensor(pic)


class LocalContrastNormalization(object):
    # implements a contrast
    def __init__(self, kernel_size=9):
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        sigma = (kernel_size - 1) / 2

        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * np.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        self.gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    def __call__(self, img_tensor):
        x = img_tensor.unsqueeze(0)  # (1, C, H, W)
        kernel = self.gaussian_kernel.to(img_tensor.device)

        local_mean = F.conv2d(x, kernel, padding=self.padding)
        x_squared = x ** 2
        local_mean_sq = F.conv2d(x_squared, kernel, padding=self.padding)
        local_var = local_mean_sq - (local_mean ** 2)
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-5))

        out = (x - local_mean) / (6 * local_std + 1e-5)
        return out.squeeze(0)


# ---------------------------------------------------------
# 2. Model Architecture
# ---------------------------------------------------------

class FourLayerResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(FourLayerResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != 256:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.bn4(self.conv4(out))
        out += residual
        out = self.relu(out)
        return out


class ExtendedEmotionDNN(nn.Module):
    def __init__(self, num_classes=6):
        super(ExtendedEmotionDNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.res1 = FourLayerResidualBlock(64)
        self.conv3 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.res2 = FourLayerResidualBlock(128)
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.flatten = nn.Flatten()
        fc_input = 512 * 8 * 6
        self.fc1 = nn.Sequential(nn.Linear(fc_input, 1024), nn.ReLU(True), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(0.5))
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.conv2(x)
        x = self.res1(x)
        x = self.pool2(self.conv3(x))
        x = self.conv4(x)
        x = self.res2(x)
        x = self.pool3(self.conv5(x))
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.output(self.fc2(self.fc1(x)))
        return x


# ---------------------------------------------------------
# 3. Data Loading
# ---------------------------------------------------------

class CKPlusDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: CSV file '{csv_file}' not found.")
            self.data = pd.DataFrame()

        self.transform = transform

        if 'emotion' in self.data.columns:
            self.data = self.data[self.data['emotion'] != 7].reset_index(drop=True)
            self.labels_list = self.data['emotion'].tolist()

        if 'Subject' not in self.data.columns:
            if 'image' in self.data.columns:
                self.data['Subject'] = self.data['image'].apply(
                    lambda x: str(x).split('_')[0] if '_' in str(x) else 'Unknown')
            else:
                self.data['Subject'] = [f"S{i // 10}" for i in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = None

        # Case 1: Load from Pixel string (CK+ cvs)
        if 'pixels' in row and isinstance(row['pixels'], str):
            try:
                pixels = np.fromstring(row['pixels'], dtype=np.float32, sep=' ').reshape(48, 48)
                pixels = pixels.astype(np.uint8)
                img = cv2.resize(pixels, (96, 128), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass

        # Case 2: Load from File Path
        if img is None and 'path' in row:
            try:
                path = row['path']
                loaded_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if loaded_img is not None:
                    img = loaded_img
            except:
                pass

        # Case 3: Fallback
        if img is None:
            img = np.zeros((128, 96), dtype=np.uint8)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        label = int(row['emotion'])

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label).long()


def get_dataloaders(csv_path, batch_size=64):
    # ADDED: FaceDetector is now the FIRST step.
    # min_size=60 ensures it skips small 48x48 pixel images automatically,
    # but catches faces in large raw images.

    val_preprocessing = T.Compose([
        FaceDetector(min_size=60),
        PaperBasedCrop(),
        SmartFeatureStitcher(),
        ToTensorAndFixDims(),
        LocalContrastNormalization()
    ])

    train_preprocessing = T.Compose([
        FaceDetector(min_size=60),
        PaperBasedCrop(),
        SmartFeatureStitcher(),
        ToTensorAndFixDims(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=5),
        LocalContrastNormalization()
    ])

    full_ds = CKPlusDataset(csv_path, transform=None)

    # Calculate Class Weights
    labels = full_ds.labels_list
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    class_weights = []

    sorted_classes = sorted(class_counts.keys())
    for c in sorted_classes:
        weight = total_samples / (num_classes * class_counts[c]) if class_counts[c] > 0 else 1.0
        class_weights.append(weight)

    weight_tensor = torch.FloatTensor(class_weights)
    print(f"Class Weights calculated: {class_weights}")

    # Subject-Independent Split
    subjects = full_ds.data['Subject'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)

    split_idx = int(len(subjects) * 0.9)
    train_subjects = subjects[:split_idx]
    test_subjects = subjects[split_idx:]

    train_indices = full_ds.data.index[full_ds.data['Subject'].isin(train_subjects)].tolist()
    test_indices = full_ds.data.index[full_ds.data['Subject'].isin(test_subjects)].tolist()

    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_ds = TransformedSubset(Subset(full_ds, train_indices), train_preprocessing)
    test_ds = TransformedSubset(Subset(full_ds, test_indices), val_preprocessing)

    print(f"Dataset Split: {len(train_ds)} Train, {len(test_ds)} Test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, weight_tensor


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csv_file_path = "ckextended_augmented.csv"

    try:
        train_loader, test_loader, class_weights = get_dataloaders(csv_file_path)
        class_weights = class_weights.to(device)

        model = ExtendedEmotionDNN(num_classes=len(class_weights)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        EPOCHS = 35
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-3)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        print("Starting Training...")
        best_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = 100. * correct / total
            train_loss = total_loss / total

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)

            val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0

            # --- Checkpointing Logic ---
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "emotion_model.pth")
                print(f"--> New Best Model Saved! (Acc: {best_acc:.2f}%)")

            scheduler.step(val_acc)

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | LR: {current_lr:.5f} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {val_acc:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()