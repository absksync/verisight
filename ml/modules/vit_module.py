from transformers import pipeline
from PIL import Image

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

def compute_vit(image_path):
    image = Image.open(image_path)
    result = classifier(image)
    return 1 - result[0]['score']