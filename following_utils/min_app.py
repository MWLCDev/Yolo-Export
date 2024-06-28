import logging
import multiprocessing
import os

from ultralytics import YOLO

quit_event = multiprocessing.Event()

model = YOLO(os.path.join(os.path.dirname(__file__), "./assets/yolov8n_800epochs(imgsz480x640).pt"))

# Declaring the logger
logging.basicConfig(format="%(levelname)s: %(asctime)s %(filename)s %(funcName)s %(message)s", datefmt="%Y%m%d:%H:%M:%S %p %Z")
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def main():

    for result in results:
        print(round(result.speed["preprocess"] + result.speed["inference"] + result.speed["postprocess"], 2))
        cls = result.boxes.cls
        for c in cls:
            pass
            # print(result.names[int(c)])


if __name__ == "__main__":

    results = model.track(source="rtsp://user1:HaikuPlot876@192.168.1.64:554/Streaming/Channels/103", show=True, stream=True, verbose=False)
    while True:
        main()
