import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset


# ---------------------------------------------------------
# 1. Preprocessing: Paper-Specific Cropping & Normalization
# ---------------------------------------------------------

class PaperBasedCrop(object):
    """
    Implements the geometric cropping described in the paper.
    (Simulated logic: in a real pipeline, this requires a facial landmark detector).
    """

    def __init__(self, output_size=(128, 96)):
        self.output_size = output_size

    def __call__(self, img):
        w, h = img.size
        # Simulated landmarks (assuming centered face)
        left_eye_center = np.array([w * 0.35, h * 0.4])
        right_eye_center = np.array([w * 0.65, h * 0.4])

        eyes_mid_point = (left_eye_center + right_eye_center) / 2
        a = np.linalg.norm(eyes_mid_point - right_eye_center)

        # Paper dimensions [Source: 109-110]
        top = int(eyes_mid_point[1] - (1.4 * a))
        bottom = int(eyes_mid_point[1] + (3.3 * a))
        left = int(eyes_mid_point[0] - (2.5 * a))
        right = int(eyes_mid_point[0] + (2.5 * a))

        top = max(0, top)
        left = max(0, left)
        bottom = min(h, bottom)
        right = min(w, right)

        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped.resize(self.output_size, Image.Resampling.BILINEAR)


class LocalContrastNormalization(object):
    """
    Implements Contrastive Equalization (Gaussian-weighted normalization).
    """

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
        x = img_tensor.unsqueeze(0)
        kernel = self.gaussian_kernel.to(img_tensor.device)

        local_mean = F.conv2d(x, kernel, padding=self.padding)
        x_squared = x ** 2
        local_mean_sq = F.conv2d(x_squared, kernel, padding=self.padding)
        local_var = local_mean_sq - (local_mean ** 2)
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-5))

        out = (x - local_mean) / (6 * local_std + 1e-5)
        return out.squeeze(0)


# ---------------------------------------------------------
# 2. Model Architecture (ExtendedEmotionDNN)
# ---------------------------------------------------------
class FourLayerResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(FourLayerResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
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
# 3. Robust Data Loading
# ---------------------------------------------------------

class CKPlusDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        if 'emotion' in self.data.columns:
            unique_labels = sorted(self.data['emotion'].unique())
            print(f"Original Emotion Labels found: {unique_labels}")

            if 7 in unique_labels:
                print("INFO: Dropping label 7 (Contempt) to keep standard 7 classes (including Disgust).")
                self.data = self.data[self.data['emotion'] != 7]

            self.data.reset_index(drop=True, inplace=True)
            final_labels = sorted(self.data['emotion'].unique())
            print(f"Final Emotion Labels used: {final_labels}")

        if 'Subject' not in self.data.columns:
            if 'image' in self.data.columns:
                print("INFO: Parsing Subject from 'image' column.")
                self.data['Subject'] = self.data['image'].apply(
                    lambda x: x.split('_')[0] if '_' in str(x) else 'Unknown')
            else:
                print("WARNING: 'Subject' column missing. Creating dummy subjects based on index.")
                self.data['Subject'] = [f"S{i // 10}" for i in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if 'pixels' in row:
            pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
            img = Image.fromarray(pixels).convert('L')
        elif 'path' in row:
            try:
                img = Image.open(row['path']).convert('L')
            except:
                img = Image.new('L', (640, 480))
        else:
            img = Image.new('L', (640, 480))

        label = int(row['emotion'])

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label).long()


def get_subject_independent_loaders(csv_path, batch_size=64):
    preprocessing = T.Compose([
        PaperBasedCrop(),
        T.Grayscale(1),
        T.Resize((128, 96)),
        T.ToTensor(),
        LocalContrastNormalization()
    ])

    full_ck_ds = CKPlusDataset(csv_path, transform=preprocessing)

    subjects = full_ck_ds.data['Subject'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)

    split_idx = int(len(subjects) * 0.9)
    train_subjects = subjects[:split_idx]
    test_subjects = subjects[split_idx:]

    train_mask = full_ck_ds.data['Subject'].isin(train_subjects)
    test_mask = full_ck_ds.data['Subject'].isin(test_subjects)

    train_indices = full_ck_ds.data.index[train_mask].tolist()
    test_indices = full_ck_ds.data.index[test_mask].tolist()

    ck_train_ds = Subset(full_ck_ds, train_indices)
    ck_test_ds = Subset(full_ck_ds, test_indices)

    print(f"Dataset Split: {len(ck_train_ds)} Train, {len(ck_test_ds)} Test")

    train_loader = DataLoader(ck_train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ck_test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        train_loader, test_loader = get_subject_independent_loaders("ckextended_augmented.csv")

        model = ExtendedEmotionDNN(num_classes=7).to(device)
        criterion = nn.CrossEntropyLoss()

        # --- REVISED SETTINGS ---
        # 1. EPOCHS: Reduced to 50 (Since convergence happens around epoch 15-20).
        EPOCHS = 50

        # 2. LR: Reduced to 0.001 (10x smaller than before to prevent crashing).
        # 3. Weight Decay: Increased to 1e-3 to prevent overfitting on the "Easy" dataset.
        optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)

        # 4. Scheduler: Adjusted to decay by 0.0001 every 10 epochs (proportional to new LR).
        # Starting LR = 0.001. After 10 epochs -> 0.0009.
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lambda epoch: max(1e-5, 1e-3 - (0.0001 * (epoch // 10))) / 1e-3)

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

            scheduler.step()

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

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | LR: {current_lr:.5f} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {val_acc:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")