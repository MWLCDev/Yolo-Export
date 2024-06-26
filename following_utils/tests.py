# This script takes .pt models as input, and runs tests on them.
# It needs to run on the nano or else will throw an error ("TensorRT model exported with a different version than 8.2.0.6")

import os
import time
import warnings

import cv2
import torch
import ultralytics
from ultralytics import YOLO

ultralytics.checks()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ModelTester:
    def __init__(self, model_names, image_filename, fixed_size=(640, 480)):
        self.model_names = model_names
        self.models_paths = {name: f"{name}.engine" for name in model_names}
        self.image_path = os.path.join(os.path.dirname(__file__), image_filename)
        self.fixed_size = fixed_size
        self.image_resized = self.preprocess_image(self.image_path, fixed_size)
        self.results = {}

    def preprocess_image(self, image_path, size):
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, dsize=size)
        return image_resized

    def measure_inference_time(self, model, image, iterations=300):
        inference_times = []

        for _ in range(iterations):
            start_time = time.time()
            model(image, imgsz=(self.fixed_size[1], self.fixed_size[0]), task="detect", verbose=True)
            end_time = time.time()
            inference_times.append(end_time - start_time)

        total_inference_time = sum(inference_times)
        average_inference_time = total_inference_time / iterations
        max_inference_time = max(inference_times)
        min_inference_time = min(inference_times)
        fps = iterations / total_inference_time
        images_per_minute = fps * 60

        return {"total_time": total_inference_time, "average_time": average_inference_time, "max_time": max_inference_time, "min_time": min_inference_time, "fps": fps, "images_per_minute": images_per_minute}

    def run_tests(self):
        for model_name in self.model_names:
            pt_model = YOLO(self.models_paths[model_name])

            print(f"\nRunning tests for {model_name}...")

            pt_performance = self.measure_inference_time(pt_model, self.image_resized)
            self.results[model_name] = pt_performance

    def display_results(self):
        for model_name, performance in self.results.items():
            print(f"\nPerformance for {model_name}.engine model:")
            print(f"Total Inference Time for 300 images: {performance['total_time']:.2f} seconds")
            print(f"Average Inference Time: {performance['average_time']:.2f} seconds")
            print(f"Max Inference Time: {performance['max_time']:.2f} seconds")
            print(f"Min Inference Time: {performance['min_time']:.2f} seconds")
            print(f"FPS: {performance['fps']:.2f}")
            print(f"Images per Minute: {performance['images_per_minute']:.2f}")


if __name__ == "__main__":
    model_names = ["yolov8n", "yolov8n_blank", "yolov8n_blank(imgsz480x640)"]
    image_filename = "bus.jpg"
    tester = ModelTester(model_names, image_filename)

    tester.run_tests()
    tester.display_results()

    time.sleep(60000)  # This number is arbitrary and just to keep the console loading for quite some time
