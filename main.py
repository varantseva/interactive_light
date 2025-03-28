import cv2
import numpy as np
from frame_preproc import preprocess_frame
from detect_module import find_largest_contour, draw_contour

CHANGE_THRESHOLD = 50

def main():
    # cap = cv2.VideoCapture(0)   #для мака
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   #для винды
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
        
        prev_area = largest_area
        prev_contour = largest_contour
 
        cv2.imshow('Line Tracking', frame)
        cv2.imshow('Binary', binary)
        
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()