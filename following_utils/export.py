import json  # Using json instead of yaml
import os
import shutil
import time
from datetime import datetime

from ultralytics import YOLO

# Get today's date in the required format
today_date = datetime.now().strftime("[%Y-%m-%dT%H:%M]")

# Define the base directory
base_dir = f"/home/ftpuser/{today_date}"


class ModelTester:
    def __init__(self, model_name, img_size=(640, 480)):
        self.img_size = img_size
        self.model_name = model_name
        self.best_model_path = os.path.join(base_dir, f"{model_name}.pt")
        self.settings_path = os.path.join(base_dir, "settings.json")  # Change to .json

    def create_model(self):
        self.pt_model = YOLO(f"./assets/{self.model_name}.pt")  # Build a new model from the .yaml configuration
        self.pt_model.predict("https://ultralytics.com/images/bus.jpg", imgsz=(self.img_size[1], self.img_size[0]))  # Predict on an image

    def export_model(self):
        # Export the model to TensorRT format with INT8 quantization
        self.pt_model.export(format="engine", device="cuda", imgsz=(self.img_size[1], self.img_size[0]), half=True)

    def move_the_model(self):
        os.makedirs(base_dir, exist_ok=True)
        # All the exported models are saved in `runs/detect/train/weights` by default
        # shutil.copy(f"runs/detect/train/weights/{self.model_name}.pt", self.best_model_path)
        shutil.move(f"./assets/{self.model_name}.engine", os.path.join(base_dir, f"{self.model_name}.engine"))
        shutil.move(f"./assets/{self.model_name}.onnx", os.path.join(base_dir, f"{self.model_name}.onnx"))
        print(f"Moved the files to {base_dir} 🤝🔥🔥")
        self.save_settings()

    def save_settings(self):
        settings = {
            "model_name": self.model_name,
            "img_size": self.img_size,
            "training_data": "coco.yaml",
            "epochs": 1,
            "classes": [0],
            "device": "cuda",
            "dynamic": True,
            "batch": -1,
        }
        with open(self.settings_path, "w") as file:
            json.dump(settings, file)
        print(f"Saved settings to {self.settings_path}")


if __name__ == "__main__":
    tester = ModelTester("yolov8_20240717_coco(imgsz480x640)")
    tester.create_model()
    tester.export_model()
    tester.move_the_model()
    time.sleep(60000)  # Arbitrary delay to simulate extended operation
