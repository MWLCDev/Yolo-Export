#bounding box

import threading
import time
from queue import Queue

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("./following_utils/yolov8n_coco(imgsz480x640_FP16_485epoch).pt",)


frame_queue = Queue()


def capture_frames(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

# https://github.com/orgs/ultralytics/discussions/10106
def process_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            results = model.predict(frame)

            # Initialize an annotator for the original frame
            annotator = Annotator(frame)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # Extract bounding box coordinates
                    c = box.cls      # Get the class of the detected object
                    annotator.box_label(b, model.names[int(c)])  # Annotate the box and class
            
            img = annotator.result()  # Get the annotated image

            # Display the image in a window named 'Annotated Image'
            cv2.imshow('Annotated Image', img)
            if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit the loop
                break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Start the frame capture and processing threads as before


rtsp_url = "rtsp://user1:HaikuPlot876@192.168.1.64:554/Streaming/Channels/103"

# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url,))
capture_thread.start()

# Start the frame processing thread
process_thread = threading.Thread(target=process_frames)
process_thread.start()

# Wait for the threads to finish
capture_thread.join()
process_thread.join()