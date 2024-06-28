#bounding box

from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture("rtsp://user1:HaikuPlot876@192.168.1.64:554/Streaming/Channels/103")

# model
model = YOLO("yolov10n.pt")

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, r.names[int(box.cls)], org, font, fontScale, color, thickness)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
