import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import os

#this code expands the dataset for both the CK+ and JAFFE dataset


def augment_csv(input_file, output_file):
    print(f"Reading {input_file}...")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return

    # Check required columns
    if 'pixels' not in df.columns and 'path' not in df.columns:
        print("Error: CSV must contain a 'pixels' column or a 'path' column.")
        return

    new_rows = []
    total_rows = len(df)

    print(f"Found {total_rows} images. Starting augmentation...")

    for index, row in df.iterrows():
        # --------------------------
        # 1. Load Image
        # --------------------------
        img = None
        if 'pixels' in row:
            try:
                # Parse "12 45 233..." string into 48x48 image
                pixel_list = np.array(row['pixels'].split(), dtype='uint8')
                # Handle cases where pixel count might not be exactly 48*48
                dim = int(np.sqrt(len(pixel_list)))
                img_array = pixel_list.reshape(dim, dim)
                img = Image.fromarray(img_array).convert('L')
            except Exception as e:
                print(f"Skipping row {index}: Bad pixel data.")
                continue
        elif 'path' in row:
            try:
                if os.path.exists(row['path']):
                    img = Image.open(row['path']).convert('L').resize((48, 48))
                else:
                    continue
            except Exception:
                continue

        if img is None:
            continue

        # Helper to convert PIL image back to CSV "pixels" string
        def img_to_str(pil_img):
            return ' '.join(map(str, np.array(pil_img).flatten()))

        # Get original metadata
        emotion = row['emotion'] if 'emotion' in row else 0
        subject = row['Subject'] if 'Subject' in row else f"S{index}"
        usage = row['Usage'] if 'Usage' in row else 'Training'

        # --------------------------
        # 2. Create Augmentations
        # --------------------------

        # Version A: Original (Just copy)
        new_rows.append({
            'emotion': emotion,
            'pixels': row['pixels'] if 'pixels' in row else img_to_str(img),
            'Usage': usage,
            'Subject': subject
        })

        # Version B: Horizontal Flip
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        new_rows.append({
            'emotion': emotion,
            'pixels': img_to_str(flipped_img),
            'Usage': usage,
            'Subject': subject
        })

        # Version C: Lighting (Brighter)
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(1.2)  # 20% brighter
        new_rows.append({
            'emotion': emotion,
            'pixels': img_to_str(bright_img),
            'Usage': usage,
            'Subject': subject
        })

        # Version D: Lighting (Darker)
        dark_img = enhancer.enhance(0.8)  # 20% darker
        new_rows.append({
            'emotion': emotion,
            'pixels': img_to_str(dark_img),
            'Usage': usage,
            'Subject': subject
        })

        if index % 100 == 0:
            print(f"Processed {index}/{total_rows} images...")

    # --------------------------
    # 3. Save
    # --------------------------
    augmented_df = pd.DataFrame(new_rows)
    augmented_df.to_csv(output_file, index=False)

    print(f"Done! Original size: {len(df)}. New size: {len(augmented_df)}.")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    augment_csv("ckextended.csv", "ckextended_augmented.csv")