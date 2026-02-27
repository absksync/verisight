from PIL import Image, ImageChops
import numpy as np
import os
import uuid

HEATMAP_DIR = r"C:/Desktop/versight_ml/ml/outputs/heatmaps"
os.makedirs(HEATMAP_DIR, exist_ok=True)

def compute_ela(image_path):
    original = Image.open(image_path).convert("RGB")
    original.save("temp.jpg", "JPEG", quality=90)

    compressed = Image.open("temp.jpg")
    ela_image = ImageChops.difference(original, compressed)

    score = np.array(ela_image).mean() / 255

    filename = str(uuid.uuid4()) + ".jpg"
    save_path = os.path.join(HEATMAP_DIR, filename)
    ela_image.save(save_path)

    return score, save_path