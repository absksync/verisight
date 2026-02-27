from PIL import Image, ImageChops
import numpy as np
import os
import uuid
import tempfile

# Save ELA visualisations under ml/outputs/heatmaps, relative to this repo.
ROOT_ML = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
HEATMAP_DIR = os.path.join(ROOT_ML, "outputs", "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)


def compute_ela(image_path):
    """
    Compute an ELA-based manipulation score in [0, 1].
    Higher scores indicate stronger compression / edit artefacts.
    """
    original = Image.open(image_path).convert("RGB")

    # On Windows, NamedTemporaryFile(delete=True) keeps the file open and
    # prevents Pillow from writing to it, so we use mkstemp instead.
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)

    try:
        original.save(tmp_path, "JPEG", quality=90)
        compressed = Image.open(tmp_path).convert("RGB")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            # If removal fails we simply leave the temp file behind.
            pass

    ela_image = ImageChops.difference(original, compressed)
    score = np.array(ela_image).mean() / 255.0

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(HEATMAP_DIR, filename)
    ela_image.save(save_path)

    return score