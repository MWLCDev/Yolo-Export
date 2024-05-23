# This script takes .pt model as input, it will export it to .engine and run tests on it
# It needs to run on the nano or else will throw error ("TensorRT model exported with a different version than 8.2.0.6")

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


def check_tensorrt():
    try:
        # Print TensorRT version
        TRT_VERSION = trt.__version__
        print(f"TensorRT version: {TRT_VERSION}")
        return True
    except ImportError as e:
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


def measure_inference_time(model, image, iterations=300):
    inference_times = []

    for i in range(iterations):
        start_time = time.time()
        model(image, imgsz=640, task="detect", verbose=True)
        end_time = time.time()
        # Calculate inference time for this iteration
        inference_time = end_time - start_time
        inference_times.append(inference_time)

    total_inference_time = sum(inference_times)
    average_inference_time = total_inference_time / iterations
    max_inference_time = max(inference_times)
    min_inference_time = min(inference_times)

    # Calculate FPS and images per minute
    fps = iterations / total_inference_time
    images_per_minute = fps * 60

    return {"total_time": total_inference_time, "average_time": average_inference_time, "max_time": max_inference_time, "min_time": min_inference_time, "fps": fps, "images_per_minute": images_per_minute}


if __name__ == "__main__":
    # Check if TensorRT is installed
    if check_tensorrt():
        # Check CUDA availability using PyTorch
        check_cuda()
        # Define paths and image URL
        model_name = "480_20k"
        pt_model_path = f"{model_name}.pt"  # Path to the .pt model file
        engine_model_path = f"/home/ftpuser/{model_name}.engine"
        image_url = "https://ultralytics.com/images/bus.jpg"
        local_image_path = download_image(image_url)
        # Read the image
        image = cv2.imread(local_image_path)
        # 480*640 h,w
        # Resize the image to a fixed size W,H (last update from M.B was the model will take a square input)
        fixed_size = (640, 480)
        image_resized = cv2.resize(image, fixed_size)

        # Export the .pt model to .engine format if not already done
        if not os.path.exists(engine_model_path):
            pt_model = YOLO(pt_model_path)
            # Export the model to the default location (this process takes a while) H,w
            pt_model.export(format="engine", device="cuda", int8=True, imgsz=480)

            # Move the exported .engine file to the FTPD directory
            default_engine_path = f"{model_name}.engine"
            if os.path.exists(default_engine_path):
                shutil.move(default_engine_path, engine_model_path)

        # Load the models
        pt_model = YOLO(pt_model_path)
        engine_model = YOLO(engine_model_path)

        pt_performance = measure_inference_time(pt_model, image_resized)
        engine_performance = measure_inference_time(engine_model, image_resized)

        # Print the results
        print(f"Image size {fixed_size}")
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
        time.sleep(60000)  # this number is arbitrary and just to keep the console loading for quite some time
