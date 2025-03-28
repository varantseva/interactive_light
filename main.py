import cv2
import numpy as np

THRESHOLD = 200
MIN_LINE_AREA = 500
MIN_LINE_WIDTH = 100
MAX_ANGLE_DEV = 5
ALERT_THRESHOLD = 50
CHANGE_THRESHOLD = 50

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


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

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return
    
    prev_contour = None
    prev_area = 0
    largest_area = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр")
            break

        prev_area = largest_area

        binary = preprocess_frame(frame)
        largest_contour, largest_area = find_largest_contour(binary)

        if prev_contour is not None:
            if abs(largest_area - prev_area) > CHANGE_THRESHOLD:
                cv2.putText(frame, "op", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 2, cv2.LINE_AA)
                frame = draw_contour(frame, largest_contour, (0, 0, 255))  # Красный цвет для изменений
            else:
                frame = draw_contour(frame, largest_contour, (0, 255, 0))  # Зеленый цвет для стабильных контуров
        else:
            frame = draw_contour(frame, largest_contour, (0, 255, 0))  # Зеленый цвет для стабильных контуров
 
        cv2.imshow('Line Tracking', frame)
        
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()