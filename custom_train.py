import sys
import os
from ultralytics import YOLO

def train_custom_yolo():
    """
    Trains a custom YOLO model with data augmentation.
    
    This script assumes it's being run in an environment like Google Colab
    where Google Drive is mounted at /content/drive.
    """
    # --- 1. Environment Setup ---
    # Add your custom ultralytics library to the Python path.
    # This ensures that your custom model layers are recognized.
    # IMPORTANT: Update these paths if your folders are in a different location in your Drive.
    # Both paths are required as custom modules are split between these two directories.
    custom_ultralytics_path_1 = '/content/drive/MyDrive/ultralytics_1'
    custom_ultralytics_path_2 = '/content/drive/MyDrive/ultralytics_2'
    
    # Check for ultralytics_1
    if not os.path.isdir(custom_ultralytics_path_1):
        print(f"ERROR: The custom ultralytics path '{custom_ultralytics_path_1}' was not found.")
        print("Please upload the 'ultralytics_1' folder to your Google Drive and update the path.")
        return
    
    # Check for ultralytics_2
    if not os.path.isdir(custom_ultralytics_path_2):
        print(f"ERROR: The custom ultralytics path '{custom_ultralytics_path_2}' was not found.")
        print("Please upload the 'ultralytics_2' folder to your Google Drive and update the path.")
        return
        
    # Add both paths to the system path. We add them at the beginning to ensure they are prioritized.
    sys.path.insert(0, custom_ultralytics_path_1)
    sys.path.insert(0, custom_ultralytics_path_2)
    print(f"Added '{custom_ultralytics_path_1}' to system path.")
    print(f"Added '{custom_ultralytics_path_2}' to system path.")

    # --- 2. Configuration ---
    # IMPORTANT: Update these paths to match the locations of your files in Google Drive.
    model_config_path = '/content/drive/MyDrive/LightYOLOv11.yaml'
    data_config_path = '/content/drive/MyDrive/usw_defect_data/data.yaml'

    if not os.path.exists(model_config_path):
        print(f"ERROR: Model config file not found at '{model_config_path}'")
        return
        
    if not os.path.exists(data_config_path):
        print(f"ERROR: Data config file not found at '{data_config_path}'")
        return

    # --- 3. Model Training ---
    print("Loading custom model from:", model_config_path)
    # Load your custom model architecture.
    # This creates a new model from your YAML definition.
    model = YOLO(model_config_path)

    print("Starting model training...")
    # Train the model with data augmentation.
    # You can adjust these augmentation parameters as needed.
    results = model.train(
        data=data_config_path,
        epochs=100,  # Adjust the number of epochs as needed
        imgsz=640,
        # --- Data Augmentation Parameters ---
        hsv_h=0.015,     # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,       # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,       # image HSV-Value augmentation (fraction)
        degrees=10.0,    # image rotation (+/- deg)
        translate=0.1,   # image translation (+/- fraction)
        scale=0.5,       # image scale (+/- gain)
        shear=2.0,       # image shear (+/- deg)
        perspective=0.0, # image perspective (+/- fraction), range 0-0.001
        flipud=0.1,      # image flip up-down (probability)
        fliplr=0.5,      # image flip left-right (probability)
        mosaic=1.0,      # mosaic augmentation (probability)
        mixup=0.1,       # mixup augmentation (probability)
        copy_paste=0.1   # copy-paste augmentation (probability)
    )
    print("Training complete.")
    print("Model and results saved in the 'runs' directory.")

if __name__ == '__main__':
    # In a Colab notebook, you would typically mount Drive first,
    # then you could run this function.
    # from google.colab import drive
    # drive.mount('/content/drive')
    # train_custom_yolo()
    pass
