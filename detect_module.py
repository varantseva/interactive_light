import cv2
import numpy as np

def find_largest_contour(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_contour = cnt

    return largest_contour, max_area

def draw_contour(frame, contour, color):
    if contour is not None:
        cv2.drawContours(frame, [contour], 0, color=color, thickness=2)
    return frame