import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from highResFrankBranchModel import crop_face_parts, FacialEmotionCNN

# --------------------------
# CONFIGURATION
# --------------------------

input_folder = "jaron_photos"   # folder of your test images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# LOAD MODEL
# --------------------------

model = FacialEmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load("highres_frankenstein.pth", map_location=device))
model.eval()

# --------------------------
# TRANSFORM (MUST MATCH TRAINING)
# --------------------------

transform = T.Compose([
    T.Resize((256, 200)),
    T.ToTensor()
])

# --------------------------
# LABEL MAPPING
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
# LOAD IMAGES
# --------------------------

files = [f for f in os.listdir(input_folder)
         if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))]

print(f"Found {len(files)} images.\n")

cropped_imgs = []
titles = []

softmax = torch.nn.Softmax(dim=1)

for fname in files:
    path = os.path.join(input_folder, fname)
    raw = cv2.imread(path)

    if raw is None:
        print(f"Could not load {fname}")
        continue

    crop = crop_face_parts(raw)
    if crop is None:
        print(f"Cropping failed for {fname}")
        continue

    pil_crop = Image.fromarray(crop)
    tensor = transform(pil_crop).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = softmax(logits)[0]
        pred_idx = torch.argmax(probs).item()
        pred_label = emotion_map[pred_idx]
        conf = probs[pred_idx].item()

    # Save for grid display
    cropped_imgs.append(crop)
    titles.append(f"{fname}\n{pred_label} ({conf*100:.1f}%)")

    print(f"{fname} → {pred_label} ({conf*100:.1f}%)")

# --------------------------
# DISPLAY ALL CROPS IN AUTO-SPACED GRID
# --------------------------

n = len(cropped_imgs)
cols = 12
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

# If there is only one row, axes may not be a list of lists
axes = np.array(axes).reshape(rows, cols)

for i, ax in enumerate(axes.flat):
    if i < n:
        ax.imshow(cropped_imgs[i], cmap="gray")
        ax.set_title(titles[i], fontsize=12)
    ax.axis("off")

plt.tight_layout(pad=1.0)   # pad = spacing between images
plt.show()
