# This script takes .pt models as input, exports them to .engine, and runs tests on them.
# It needs to run on the nano or else will throw an error ("TensorRT model exported with a different version than 8.2.0.6")

import os
import shutil
import subprocess
import time
import warnings

import cv2
import tensorrt as trt
import torch
import ultralytics
from ultralytics import YOLO

ultralytics.checks()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ModelTester:
    def __init__(self, model_names, image_url, fixed_size=(640, 480)):
        self.model_names = model_names
        self.models_paths = {name: {"pt": f"{name}.pt", "engine": f"/home/ftpuser/{name}.engine"} for name in model_names}
        self.image_url = image_url
        self.fixed_size = fixed_size
        self.local_image_path = self.download_image(image_url)
        self.image_resized = self.preprocess_image(self.local_image_path, fixed_size)

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

    def download_image(image_url, local_path="bus.jpg"):
        if not os.path.exists(local_path):
            subprocess.run(["wget", image_url, "-O", local_path])
        return local_path

    def preprocess_image(image_path, size):
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, size)
        return image_resized

    def measure_inference_time(model, image, iterations=300):
        inference_times = []

        for _ in range(iterations):
            start_time = time.time()
            model(image, imgsz=640, task="detect", verbose=True)
            end_time = time.time()
            inference_times.append(end_time - start_time)

        total_inference_time = sum(inference_times)
        average_inference_time = total_inference_time / iterations
        max_inference_time = max(inference_times)
        min_inference_time = min(inference_times)
        fps = iterations / total_inference_time
        images_per_minute = fps * 60

        return {"total_time": total_inference_time, "average_time": average_inference_time, "max_time": max_inference_time, "min_time": min_inference_time, "fps": fps, "images_per_minute": images_per_minute}

    def export_model(self, model_name):
        pt_model_path = self.models_paths[model_name]["pt"]
        engine_model_path = self.models_paths[model_name]["engine"]
        
        if not os.path.exists(engine_model_path):
            pt_model = YOLO(pt_model_path)
            # (you might want to change imgsz parameter here)
            pt_model.export(format="engine", device="cuda", int8=True, imgsz=640)

            default_engine_path = f"{model_name}.engine"
            if os.path.exists(default_engine_path):
                shutil.move(default_engine_path, engine_model_path)

    def run_tests(self):
        for model_name in self.model_names:
            pt_model = YOLO(self.models_paths[model_name]["pt"])
            engine_model = YOLO(self.models_paths[model_name]["engine"])

            print(f"\nRunning tests for {model_name}...")

            pt_performance = self.measure_inference_time(pt_model, self.image_resized)
            engine_performance = self.measure_inference_time(engine_model, self.image_resized)

            print(f"\nPerformance for {model_name} .engine model:")
            print(f"Total Inference Time for 300 images: {engine_performance['total_time']:.2f} seconds")
            print(f"Average Inference Time: {engine_performance['average_time']:.2f} seconds")
            print(f"Max Inference Time: {engine_performance['max_time']:.2f} seconds")
            print(f"Min Inference Time: {engine_performance['min_time']:.2f} seconds")
            print(f"FPS: {engine_performance['fps']:.2f}")
            print(f"Images per Minute: {engine_performance['images_per_minute']:.2f}")

            print(f"\nPerformance for {model_name} .pt model:")
            print(f"Total Inference Time for 300 images: {pt_performance['total_time']:.2f} seconds")
            print(f"Average Inference Time: {pt_performance['average_time']:.2f} seconds")
            print(f"Max Inference Time: {pt_performance['max_time']:.2f} seconds")
            print(f"Min Inference Time: {pt_performance['min_time']:.2f} seconds")
            print(f"FPS: {pt_performance['fps']:.2f}")
            print(f"Images per Minute: {pt_performance['images_per_minute']:.2f}")


if __name__ == "__main__":
    model_names = ["yolov8n", "yolov10n"]
    image_url = "https://ultralytics.com/images/bus.jpg"
    tester = ModelTester(model_names, image_url)

    if tester.check_tensorrt():
        tester.check_cuda()
        for model_name in model_names:
            tester.export_model(model_name)
        tester.run_tests()

    time.sleep(60000)  # This number is arbitrary and just to keep the console loading for quite some time
