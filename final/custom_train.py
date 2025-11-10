import sys
import os
from ultralytics import YOLO

def train_custom_yolo():
    """
    Trains a custom YOLO model with data augmentation.
    
    This script is designed to be self-contained. It assumes it is located
    in a folder that also contains the 'ultralytics' library folder and the model .yaml file.
    """
    # --- 1. Environment Setup ---
    # This script will automatically locate the 'ultralytics' library
    # in the same directory as the script itself.
    try:
        # Get the directory where this script is located.
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for environments like notebooks where __file__ is not defined
        script_dir = '.'

    custom_ultralytics_path = os.path.join(script_dir, 'ultralytics')
    
    if not os.path.isdir(custom_ultralytics_path):
        print(f"ERROR: The 'ultralytics' directory was not found in the same folder as the script.")
        print(f"Looked for: {os.path.abspath(custom_ultralytics_path)}")
        return
        
    # Add the local ultralytics path to the system path.
    sys.path.insert(0, custom_ultralytics_path)
    print(f"Added '{custom_ultralytics_path}' to system path.")

    # --- 2. Configuration ---
    # The model config is now local to this folder.
    model_config_path = os.path.join(script_dir, 'LightYOLOv11.yaml')
    
    # IMPORTANT: The dataset path still needs to point to your data on Google Drive.
    # Update this path if your dataset is in a different location.
    data_config_path = '/content/drive/MyDrive/usw_defect_data/data.yaml'

    if not os.path.exists(model_config_path):
        print(f"ERROR: Model config file not found at '{model_config_path}'")
        return
        
    if not os.path.exists(data_config_path):
        print(f"ERROR: Data config file not found at '{data_config_path}'")
        print("Please ensure your dataset is in Google Drive and the path is correct.")
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
