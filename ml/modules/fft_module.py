import cv2
import numpy as np

def compute_fft(image_path):
    img = cv2.imread(image_path, 0)
    f = np.fft.fft2(img)
    magnitude = np.log(np.abs(f) + 1)
    score = np.mean(magnitude) / 10
    return min(score, 1.0)