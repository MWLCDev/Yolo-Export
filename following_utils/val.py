from ultralytics import YOLO

# Load the model
model = YOLO("./following_utils/assets/yolov8_20240717_coco(imgsz480x640).pt")

# Validate the model
metrics = model.val(data="coco.yaml", imgsz=(480, 640))  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
