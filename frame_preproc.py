import cv2
import numpy as np

THRESHOLD = 220

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)