import os
import cv2
import time
from ultralytics import YOLO

save_dir = "Media"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

model = YOLO('yolo11n.pt')

if not cap.isOpened():
    print('No Camera')
    exit()

counter = 0


while True:

    start_fps = time.time()

    ret, frame = cap.read()


    if not ret:
        print("No Frame")
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:

            conf = float(box.conf[0])
            cls = int(box.cls[0])
        labels = model.names[cls]


        if labels == 'person':

            end_fps = time.time()
            start_fps = end_fps
            FPS = 1 / (start_fps - end_fps)

            cv2.putText(
                frame,
                f'FPS: {FPS}',
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f'{labels}',
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                'MotionsWeb',
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )



    cv2.imshow('Video With YOLO', frame)

    counter += 1

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()