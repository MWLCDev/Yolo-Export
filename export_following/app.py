# This script takes a .pt model as input, exports it to .engine, and runs tests on it.
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
    def __init__(self, model_name, image_url, fixed_size=(640, 480)):
        """
        Initialize the ModelTester with the model name, image URL, and fixed size for the image.

        :param model_name: Name of the model file without extension.
        :param image_url: URL to download the test image.
        :param fixed_size: Fixed size (width, height) to resize the image to.
        """
        self.model_name = model_name
        self.pt_model_path = f"{model_name}.pt"
        self.engine_model_path = f"/home/ftpuser/{model_name}.engine"
        self.image_url = image_url
        self.fixed_size = fixed_size
        self.local_image_path = self.download_image(image_url)
        self.image_resized = self.preprocess_image(self.local_image_path, fixed_size)

    @staticmethod
    def check_tensorrt():
        """
        Check if TensorRT is installed and print its version.

        :return: True if TensorRT is installed, False otherwise.
        """
        try:
            TRT_VERSION = trt.__version__
            print(f"TensorRT version: {TRT_VERSION}")
            return True
        except ImportError:
            print("TensorRT is not installed.")
            return False

    @staticmethod
    def check_cuda():
        """
        Check if CUDA is available and print its version.
        """
        if torch.cuda.is_available():
            print(f"CUDA is available. CUDA version: {torch.version.cuda}")
        else:
            print("CUDA is not available.")

    @staticmethod
    def download_image(image_url, local_path="bus.jpg"):
        """
        Download an image from a URL to a local path.

        :param image_url: URL of the image to download.
        :param local_path: Local path to save the downloaded image.
        :return: Path to the downloaded image.
        """
        if not os.path.exists(local_path):
            subprocess.run(["wget", image_url, "-O", local_path])
        return local_path

    @staticmethod
    def preprocess_image(image_path, size):
        """
        Read and resize the image to the specified size.

        :param image_path: Path to the image file.
        :param size: Tuple specifying the size (width, height) to resize the image to.
        :return: Resized image.
        """
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, size)
        return image_resized

    @staticmethod
    def measure_inference_time(model, image, iterations=300):
        """
        Measure the inference time of a model over a specified number of iterations.

        :param model: Model to be tested.
        :param image: Input image for the model.
        :param iterations: Number of iterations to measure.
        :return: Dictionary containing inference time statistics.
        """
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

    def export_model(self):
        """
        Export the .pt model to .engine format if not already done.
        """
        if not os.path.exists(self.engine_model_path):
            pt_model = YOLO(self.pt_model_path)
            # (you might want to change imgsz parameter here)
            pt_model.export(format="engine", device="cuda", int8=True, imgsz=640)

            default_engine_path = f"{self.model_name}.engine"
            if os.path.exists(default_engine_path):
                shutil.move(default_engine_path, self.engine_model_path)

    def run_tests(self):
        """
        Run inference tests on both the .pt and .engine models and print the performance results.
        """
        pt_model = YOLO(self.pt_model_path)
        engine_model = YOLO(self.engine_model_path)

        pt_performance = self.measure_inference_time(pt_model, self.image_resized)
        engine_performance = self.measure_inference_time(engine_model, self.image_resized)

        print(f"Image size {self.fixed_size}")
        print("Performance for .engine model:")
        print(f"Total Inference Time for 300 images: {engine_performance['total_time']:.2f} seconds")
        print(f"Average Inference Time: {engine_performance['average_time']:.2f} seconds")
        print(f"Max Inference Time: {engine_performance['max_time']:.2f} seconds")
        print(f"Min Inference Time: {engine_performance['min_time']:.2f} seconds")
        print(f"FPS: {engine_performance['fps']:.2f}")
        print(f"Images per Minute: {engine_performance['images_per_minute']:.2f}")

        print("\nPerformance for .pt model:")
        print(f"Total Inference Time for 300 images: {pt_performance['total_time']:.2f} seconds")
        print(f"Average Inference Time: {pt_performance['average_time']:.2f} seconds")
        print(f"Max Inference Time: {pt_performance['max_time']:.2f} seconds")
        print(f"Min Inference Time: {pt_performance['min_time']:.2f} seconds")
        print(f"FPS: {pt_performance['fps']:.2f}")
        print(f"Images per Minute: {pt_performance['images_per_minute']:.2f}")


if __name__ == "__main__":
    model_name = "yolov8n"
    image_url = "https://ultralytics.com/images/bus.jpg"
    tester = ModelTester(model_name, image_url)

    if tester.check_tensorrt():
        tester.check_cuda()
        tester.export_model()
        tester.run_tests()

    time.sleep(60000)  # This number is arbitrary and just to keep the console loading for quite some time
