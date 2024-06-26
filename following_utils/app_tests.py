# run tests without exports
import cv2
import time
from ultralytics import YOLO

# Define paths
engine_model_path = "yolov8n.engine"  # Path to the .engine model file
pt_model_path = "yolov8n.pt"  # Path to the .pt model file
image_path = "zidane.jpg"  # Path to the image file

# Initialize the models
engine_model = YOLO(engine_model_path)
pt_model = YOLO(pt_model_path)

# Read the image
image = cv2.imread(image_path)

# Resize the image to a fixed size (last update from MB was the model will take a square input)
fixed_size = (640, 640)
image_resized = cv2.resize(image, fixed_size)


def measure_inference_time(model, image, iterations=300):
    inference_times = []

    for i in range(iterations):
        start_time = time.time()
        model(image, imgsz=640, verbose=True, task="detect")
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


# Measure inference time for .pt model
pt_performance = measure_inference_time(pt_model, image)

# Measure inference time for .engine model
engine_performance = measure_inference_time(engine_model, image)

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
