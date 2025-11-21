import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torchvision.transforms as T
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


# ---------------------------------------------------------
# 1. Preprocessing: Local Contrast Normalization (LCN)
# ---------------------------------------------------------
class LocalContrastNormalization(object):
    """
    Implements the Contrastive Equalization described in Section 3.2.
    Formula: (Image - Local Mean) / Local Std Dev
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

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
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
# 2. Model Architecture
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
        # Conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Res1
        self.res1 = FourLayerResidualBlock(in_channels=64)
        # Conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Res2
        self.res2 = FourLayerResidualBlock(in_channels=128)
        # Conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()

        fc_input_dim = 512 * 8 * 6
        self.fc1 = nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.output = nn.Linear(512, num_classes)

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
        x = self.output(x)
        return x


# ---------------------------------------------------------
# 3. Data & Training
# ---------------------------------------------------------

train_transforms = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((128, 96)),
    T.ToTensor(),
    LocalContrastNormalization(kernel_size=9)
])


class GenericEmotionDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Mock image loading
        img = Image.new('L', (96, 128))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label).long()


def get_dataloaders(ck_csv_path, jaffe_folder_path):
    # Mock data setup
    jaffe_total_imgs = ["img"] * 213
    jaffe_total_lbls = [0] * 213
    jaffe_full_ds = GenericEmotionDataset(jaffe_total_imgs, jaffe_total_lbls, transform=train_transforms)

    jaffe_train, jaffe_test = random_split(
        jaffe_full_ds, [200, 13],
        generator=torch.Generator().manual_seed(42)
    )

    ck_total_imgs = ["img"] * 8150
    ck_total_lbls = [0] * 8150
    ck_full_ds = GenericEmotionDataset(ck_total_imgs, ck_total_lbls, transform=train_transforms)

    ck_train, ck_test = random_split(
        ck_full_ds, [8000, 150],
        generator=torch.Generator().manual_seed(42)
    )

    combined_train_ds = ConcatDataset([jaffe_train, ck_train])

    train_loader = DataLoader(combined_train_ds, batch_size=64, shuffle=True)
    ck_test_loader = DataLoader(ck_test, batch_size=64, shuffle=False)

    return train_loader, ck_test_loader


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ExtendedEmotionDNN(num_classes=6).to(device)
    optimizer = optim.SGD(model.parameters(), lr=4e-2, weight_decay=1e-5)


    def lr_lambda(epoch):
        initial_lr = 4e-2
        drop = 0.005
        epochs_drop = 10
        decay_steps = epoch // epochs_drop
        new_lr = initial_lr - (drop * decay_steps)
        return max(new_lr, 1e-5) / initial_lr


    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    train_loader, ck_test_loader = get_dataloaders("ck.csv", "jaffe_folder")

    EPOCHS = 50

    for epoch in range(EPOCHS):
        model.train()

        # --- Restore original metrics tracking ---
        total_loss = 0
        correct = 0
        total = 0
        # ------------------------------------------

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # --- Restore calculation logic ---
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # ---------------------------------

        scheduler.step()

        # Calculate metrics exactly as before
        train_loss = total_loss / total
        train_acc = 100 * correct / total

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in ck_test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"CK+ Validation Accuracy: {val_acc:.2f}%")