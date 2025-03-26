import cv2
import numpy as np

THRESHOLD = 200
MIN_LINE_AREA = 500
MIN_LINE_WIDTH = 100
MAX_ANGLE_DEV = 5
ALERT_THRESHOLD = 50

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

def detect_dominant_line(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_line = None
    contur = None
    max_width = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_LINE_AREA:
            continue
            
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        current_width = max(width, height)
        
        if current_width < MIN_LINE_WIDTH:
            continue
            
        angle = rect[2]
        if not (-MAX_ANGLE_DEV <= angle <= MAX_ANGLE_DEV):
            continue
            
        if current_width > max_width:
            max_width = current_width
            best_line = rect
            contur = cnt

    
    return best_line, contur

def draw_center_point(frame, rect, ctr):
    if rect is None:
        return frame
        
    try:
        center_x, center_y = rect[0]

        center = (int(center_x), int(center_y))
        
        cv2.circle(frame, center, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.drawContours(frame, [ctr], -1, color=(255, 255, 0), thickness=-1)
        
    except Exception as e:
        print(f"Ошибка при отрисовке центра: {e}")
    
    return frame

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return
    
    prev_line = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр")
            break

        binary = preprocess_frame(frame)
        contur = None
        line_rect, contur = detect_dominant_line(binary)

        frame = draw_center_point(frame, line_rect, contur)

        if line_rect is not None:
            center_x = int(line_rect[0][0])
            frame_center = frame.shape[1] // 2
            
            if abs(center_x - frame_center) > ALERT_THRESHOLD:
                cv2.putText(frame, "ALERT: Line Shifted!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Line Detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Line Tracking', frame)
        
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()