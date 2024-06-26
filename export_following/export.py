# This script takes a .pt model as input then exports it to .engine.
# It needs to run on the nano or else will throw an error ("TensorRT model exported with a different version than 8.2.0.6")

import os
import shutil
import time
import warnings

import tensorrt as trt
import torch
import ultralytics
from ultralytics import YOLO

ultralytics.checks()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ModelTester:
    def __init__(self, model_name):

        self.model_name = model_name
        self.pt_model_path = f"{model_name}.pt"
        self.engine_model_path = f"/home/ftpuser/{model_name}.engine"

    def check_tensorrt():
        try:
            TRT_VERSION = trt.__version__
            print(f"TensorRT version: {TRT_VERSION}")
            return True
        except ImportError:
            print("TensorRT is not installed.")
            return False

    def check_cuda():
        if torch.cuda.is_available():
            print(f"CUDA is available. CUDA version: {torch.version.cuda}")
        else:
            print("CUDA is not available.")

    def export_model(self):

        if not os.path.exists(self.engine_model_path):
            pt_model = YOLO(self.pt_model_path)
            # (you might want to change imgsz parameter here)
            pt_model.export(format="engine", device="cuda", int8=True, imgsz=640)

            default_engine_path = f"{self.model_name}.engine"
            if os.path.exists(default_engine_path):
                shutil.move(default_engine_path, self.engine_model_path)


if __name__ == "__main__":
    model_name = "yolov10n"
    tester = ModelTester(model_name)

    if tester.check_tensorrt():
        tester.check_cuda()
        tester.export_model()

    time.sleep(60000)  # This number is arbitrary and just to keep the console loading for quite some time
