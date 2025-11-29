import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from highResFrankenstein import crop_eyes_mouth_vertical, FacialEmotionCNN

# --------------------------
# CONFIGURATION
# --------------------------

input_folder = "jaron_photos"  # folder with all your test images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# LOAD MODEL
# --------------------------

model = FacialEmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load("highres_frankenstein.pth", map_location=device))
model.eval()

# --------------------------
# IMAGE TRANSFORM (must match training)
# --------------------------

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# --------------------------
# EMOTION LABELS
# --------------------------

emotion_map = {
    0: "AN",  # Angry
    1: "DI",  # Disgust
    2: "FE",  # Fear
    3: "HA",  # Happy
    4: "NE",  # Neutral
    5: "SA",  # Sad
    6: "SU"   # Surprise
}

# --------------------------
# PROCESS ALL IMAGES IN FOLDER
# --------------------------

files = [f for f in os.listdir(input_folder)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))]

print(f"Found {len(files)} images to test.\n")

for fname in files:
    path = os.path.join(input_folder, fname)
    raw = cv2.imread(path)

    if raw is None:
        print(f"[ERROR] Could not load {fname}")
        continue

    # crop eyes + mouth vertically stacked
    crop = crop_eyes_mouth_vertical(raw)
    if crop is None:
        print(f"[ERROR] Cropping failed for {fname}")
        continue

    pil_crop = Image.fromarray(crop)

    img_tensor = transform(pil_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_idx = pred_logits.argmax(dim=1).item()
        pred_label = emotion_map[pred_idx]

    print(f"{fname} → Predicted Emotion: {pred_label}")

print("\nDone predicting all images.")
