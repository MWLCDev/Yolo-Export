# # This script takes a .yaml model configuration file as input, creates a model,
# # saves it as a .pt file, and then exports it to a .engine file.
# # It needs to run on the nano or else will throw an error
# # ("TensorRT model exported with a different version than 8.2.0.6")
import os
import shutil
import time
from datetime import datetime

from ultralytics import YOLO

# Get today's date in the required format
today_date = datetime.now().strftime("[%Y-%m-%dT%H:%M]")

# Define the base directory
base_dir = f"/home/ftpuser/{today_date}"
os.makedirs(base_dir, exist_ok=True)

class ModelTester:
    def __init__(self, model_name, img_size=(640, 480)):
        self.img_size = img_size
        self.model_name = model_name
        self.engine_model_path = os.path.join(base_dir, f"{model_name}.engine")
        self.best_model_path = os.path.join(base_dir, "best.pt")

    def create_model(self):
        self.pt_model = YOLO(f"./{self.model_name}.yaml")  # build a new model from the .yaml configuration
        self.pt_model.train(data="coco8.yaml", epochs=1, imgsz=(self.img_size[1], self.img_size[0]), classes=[0], device="cuda")
        self.pt_model.val()  # evaluate model performance on the validation set
        self.pt_model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Copy the best.pt model to the base directory
        shutil.copy("runs/detect/train/weights/best.pt", self.best_model_path)
        print(f"Copied best.pt to {self.best_model_path}")

    def export_model(self):
        if not os.path.exists(self.engine_model_path):
            # Export the model to TensorRT format with INT8 quantization
            engine_path = self.pt_model.export(format="engine", device="cuda", imgsz=(self.img_size[1], self.img_size[0]))
            shutil.move(engine_path, self.engine_model_path)
            print(f"Moved the file to {self.engine_model_path}🤝🔥🔥")

if __name__ == "__main__":
    tester = ModelTester("yolov8n")

    tester.create_model()
    tester.export_model()

    time.sleep(60000)  # This number is arbitrary and just to keep the console loading for quite some time
