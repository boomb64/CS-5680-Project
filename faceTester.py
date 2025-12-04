import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math


# ---------------------------------------------------------
# 1. NEW: Face Detection Class (The Missing Link)
# ---------------------------------------------------------
class FaceDetector(object):
    """
    Finds the face in a large image and crops to it.
    Essential for processing raw photos that aren't already cropped.
    """

    def __init__(self, padding=0):
        try:
            # Load the standard frontal face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.active = True
        except:
            print("Warning: Frontal face cascade not found. Skipping face detection.")
            self.active = False
        self.padding = padding

    def __call__(self, img):
        if not self.active:
            return img

        # Detect faces
        faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            # If no face found, return original (fallback)
            return img

        # Assume the largest face is the target
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        # Add padding if desired (careful not to go out of bounds)
        # CK+ is usually very tight, so minimal padding is best
        return img[y:y + h, x:x + w]


# ---------------------------------------------------------
# 2. Existing Preprocessing (Must match training)
# ---------------------------------------------------------

class PaperBasedCrop(object):
    """
    Simulates the paper's geometric cropping.
    """

    def __init__(self, output_size=(128, 96)):
        self.target_size = (96, 128)  # W, H for cv2.resize

    def __call__(self, img):
        h, w = img.shape[:2]
        # Estimate landmarks for a generally centered face
        left_eye_center = np.array([w * 0.35, h * 0.4])
        right_eye_center = np.array([w * 0.65, h * 0.4])
        eyes_mid_point = (left_eye_center + right_eye_center) / 2
        a = np.linalg.norm(eyes_mid_point - right_eye_center)

        top = int(eyes_mid_point[1] - (1.4 * a))
        bottom = int(eyes_mid_point[1] + (3.3 * a))
        left = int(eyes_mid_point[0] - (2.5 * a))
        right = int(eyes_mid_point[0] + (2.5 * a))

        top = max(0, top)
        left = max(0, left)
        bottom = min(h, bottom)
        right = min(w, right)

        img_cropped = img[top:bottom, left:right]

        if img_cropped.size == 0:
            return cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        return cv2.resize(img_cropped, self.target_size, interpolation=cv2.INTER_LINEAR)


class SmartFeatureStitcher(object):
    """
    Tries to find eyes/mouth using Haar Cascades (Smart).
    Falls back to fixed percentages if detection fails.
    """

    def __init__(self, output_size=(96, 128)):
        self.output_size = output_size
        try:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            self.use_smart = True
        except:
            print("Warning: Haar cascades not found. Using fixed percentages.")
            self.use_smart = False

    def __call__(self, img):
        h, w = img.shape[:2]
        eyes_strip = None
        mouth_strip = None

        if self.use_smart:
            top_half = img[:int(h * 0.6), :]
            eyes = self.eye_cascade.detectMultiScale(top_half, 1.1, 5, minSize=(20, 20))

            bottom_half = img[int(h * 0.5):, :]
            mouths = self.mouth_cascade.detectMultiScale(bottom_half, 1.1, 5, minSize=(25, 15))

            if len(eyes) >= 1:
                min_y = np.min(eyes[:, 1])
                max_y_h = np.max(eyes[:, 1] + eyes[:, 3])
                y1 = max(0, min_y - 10)
                y2 = min(int(h * 0.6), max_y_h + 10)
                eyes_strip = img[y1:y2, :]

            if len(mouths) >= 1:
                mouths = sorted(mouths, key=lambda x: x[2] * x[3], reverse=True)
                mx, my, mw, mh = mouths[0]
                real_y = my + int(h * 0.5)
                y1 = max(int(h * 0.5), real_y - 10)
                y2 = min(h, real_y + mh + 10)
                mouth_strip = img[y1:y2, :]

        if eyes_strip is None or eyes_strip.size == 0:
            eyes_strip = img[int(h * 0.15):int(h * 0.50), :]

        if mouth_strip is None or mouth_strip.size == 0:
            mouth_strip = img[int(h * 0.65):int(h * 0.95), :]

        try:
            combined_img = np.vstack((eyes_strip, mouth_strip))
        except ValueError:
            return cv2.resize(img, self.output_size, interpolation=cv2.INTER_LINEAR)

        return cv2.resize(combined_img, self.output_size, interpolation=cv2.INTER_LINEAR)


class ToTensorAndFixDims(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            if len(pic.shape) == 2:
                pic = pic[:, :, None]
            return T.functional.to_tensor(pic)
        return T.functional.to_tensor(pic)


class LocalContrastNormalization(object):
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
# 3. Model Architecture
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
    def __init__(self, num_classes=7):
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
        self.fc1 = nn.Sequential(nn.Linear(fc_input, 1024), nn.ReLU(True), nn.Dropout(0.6))
        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(0.6))
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
# 4. Prediction Loop
# ---------------------------------------------------------

EMOTION_MAP = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise",
    6: "Neutral"
}


def predict_folder(folder_path="finalTest", model_path="emotion_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ExtendedEmotionDNN(num_classes=7).to(device)

    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model file '{model_path}' not found.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    # --- UPDATED PIPELINE ---
    # 1. Detect Face in large photo (NEW)
    face_detector = FaceDetector()

    # 2. Structure (Crop + Stitch) -> Returns Numpy Image for Display
    structural_preprocess = T.Compose([
        face_detector,  # Find face first!
        PaperBasedCrop(),
        SmartFeatureStitcher(),
    ])

    # 3. Tensor (ToTensor + Normalize) -> Returns Tensor for AI
    tensor_preprocess = T.Compose([
        ToTensorAndFixDims(),
        LocalContrastNormalization()
    ])

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if len(image_files) == 0:
        print(f"No image files found in '{folder_path}'")
        return

    print(f"\nFound {len(image_files)} images in '{folder_path}'. Running inference...\n")

    results = []

    for img_file in image_files:
        full_path = os.path.join(folder_path, img_file)
        raw_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if raw_img is None:
            continue

        try:
            # Step A: Get the "Frankenstein" image
            # Now includes Face Detection -> Crop -> Stitch
            stitched_img = structural_preprocess(raw_img)

            # Step B: Prepare for Model
            input_tensor = tensor_preprocess(stitched_img)
            input_batch = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_batch)
                probabilities = F.softmax(outputs, dim=1)

                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_class = EMOTION_MAP.get(predicted_idx.item(), "Unknown")
                conf_score = confidence.item() * 100

                print(f"{img_file:<30} | {predicted_class:<12} | {conf_score:.2f}%")

                label_text = f"{predicted_class}\n({conf_score:.1f}%)"
                results.append((stitched_img, label_text, img_file))

        except Exception as e:
            print(f"{img_file:<30} | Error during processing: {e}")

    # Display Grid
    if len(results) > 0:
        num_images = len(results)
        cols = 4
        rows = math.ceil(num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))
        fig.suptitle("Model Predictions on Frankstein Inputs", fontsize=16)

        axes = axes.flatten() if num_images > 1 else [axes]

        for i, (img, label, fname) in enumerate(results):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(label, fontsize=12, color='blue', fontweight='bold')
            axes[i].set_xlabel(fname, fontsize=8)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    folder_to_test = "finalTest"
    predict_folder(folder_path=folder_to_test)