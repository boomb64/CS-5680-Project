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



# the two main changes implemented are expanding the dataset by transoforming
# the provided dataset. the other main change is altering the images by
# removing all areas of the face that aren't the face and mouth
# this code does the face removing and putting the images through the CNN

#preprocessing the images

class PaperBasedCrop(object):
    #does the cropping

    def __init__(self, output_size=(128, 96)):
        # getting dimensions fromt he paper
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


#this class is my original addition. It completely removes all the other parts instead
#of just warping the shape of the face
class SalientFeatureStitcher(object):
    #cuts out the eyes, eyebrows, and mouth and gets rid of everything else

    def __init__(self, output_size=(96, 128)):
        self.output_size = output_size  # (Width, Height) for cv2

    def __call__(self, img):
        # img is numpy array (H, W)
        h, w = img.shape[:2]

        # Define slice percentages for a standard aligned face
        # Top Strip (Eyebrows + Eyes): ~Top 15% to 50%
        eye_top = int(h * 0.15)
        eye_bottom = int(h * 0.50)

        # Bottom Strip (Mouth): ~Top 65% to 95%
        mouth_top = int(h * 0.65)
        mouth_bottom = int(h * 0.95)

        # Slicing
        eyes_strip = img[eye_top:eye_bottom, :]
        mouth_strip = img[mouth_top:mouth_bottom, :]

        # Stitching (Vertical concatenation)
        try:
            combined_img = np.vstack((eyes_strip, mouth_strip))
        except ValueError:
            # Fallback if dimensions mismatch slightly
            return cv2.resize(img, self.output_size, interpolation=cv2.INTER_LINEAR)

        # Resize back to model input size
        return cv2.resize(combined_img, self.output_size, interpolation=cv2.INTER_LINEAR)


class ToTensorAndFixDims(object):
    #converts to a tensor
    def __call__(self, pic):
        # pic is numpy array
        if isinstance(pic, np.ndarray):
            # If (H, W), expand to (H, W, 1) for ToTensor compatibility
            if len(pic.shape) == 2:
                pic = pic[:, :, None]

            # ToTensor converts (H, W, C) [0, 255] -> (C, H, W) [0.0, 1.0]
            return T.functional.to_tensor(pic)
        return T.functional.to_tensor(pic)


class LocalContrastNormalization(object):
    #implements a contrast
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
        # img_tensor shape: (C, H, W)
        x = img_tensor.unsqueeze(0)  # (1, C, H, W)
        kernel = self.gaussian_kernel.to(img_tensor.device)

        local_mean = F.conv2d(x, kernel, padding=self.padding)
        x_squared = x ** 2
        local_mean_sq = F.conv2d(x_squared, kernel, padding=self.padding)
        local_var = local_mean_sq - (local_mean ** 2)
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-5))

        out = (x - local_mean) / (6 * local_std + 1e-5)
        return out.squeeze(0)


#the actual cnn arcticture, from the paper
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


#loading the data
class CKPlusDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: CSV file '{csv_file}' not found.")
            self.data = pd.DataFrame({
                'emotion': np.random.randint(0, 6, 10),
                'pixels': [' '.join(['0'] * 2304) for _ in range(10)],
                'Subject': [f'S{i}' for i in range(10)]
            })

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
                # Convert to uint8 for cv2 processing if needed, though float is okay.
                # Standardize to 0-255 uint8 for consistent cv2 behavior
                pixels = pixels.astype(np.uint8)
                # Resize to target face size used in this pipeline (W=96, H=128)
                # Note: Paper uses H=128, W=96
                img = cv2.resize(pixels, (96, 128), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass

        # Case 2: Load from File Path
        if img is None and 'path' in row:
            try:
                path = row['path']
                # Read as Grayscale
                loaded_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if loaded_img is not None:
                    img = loaded_img
            except:
                pass

        # Case 3: Fallback
        if img is None:
            img = np.zeros((128, 96), dtype=np.uint8)

        # Ensure image is uint8 before transforms (cv2 standard)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        label = int(row['emotion'])

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label).long()


def get_dataloaders(csv_path, batch_size=64):
    # set up training and validation with cv2
    # Validation Transform
    val_preprocessing = T.Compose([
        PaperBasedCrop(),  # Returns numpy (H, W)
        SalientFeatureStitcher(),  # Returns numpy (H, W)
        ToTensorAndFixDims(),  # Returns Tensor (1, H, W)
        LocalContrastNormalization()  # Operates on Tensor
    ])

    # Training Transform
    train_preprocessing = T.Compose([
        PaperBasedCrop(),  # Returns numpy (H, W)
        SalientFeatureStitcher(),  # Returns numpy (H, W)
        ToTensorAndFixDims(),  # Convert to Tensor for easy augmentation
        T.RandomHorizontalFlip(p=0.5),  # Works on Tensors
        T.RandomRotation(degrees=5),  # Works on Tensors
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

        EPOCHS = 50
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-3)

        # Corrected line: removed verbose=True
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        print("Starting Training...")
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

            scheduler.step(val_acc)

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | LR: {current_lr:.5f} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {val_acc:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()