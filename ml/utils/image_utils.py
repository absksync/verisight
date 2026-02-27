import cv2
import os
import uuid

HIGHLIGHT_DIR = r"C:/Users/Apoorva/Desktop/VERISIGHT/ml/outputs/highlighted"
os.makedirs(HIGHLIGHT_DIR, exist_ok=True)

def highlight_region(image_path, bbox):
    if bbox is None:
        return None

    img = cv2.imread(image_path)

    x1 = int(bbox[0][0])
    y1 = int(bbox[0][1])
    x2 = int(bbox[2][0])
    y2 = int(bbox[2][1])

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

    filename = str(uuid.uuid4()) + ".jpg"
    save_path = os.path.join(HIGHLIGHT_DIR, filename)
    cv2.imwrite(save_path, img)

    return save_path