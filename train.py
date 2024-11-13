import os
from pathlib import Path
from ultralytics import YOLO
import yaml

def train_and_export():
    # Create dataset config
    current_dir = os.getcwd()
    dataset_config = {
        'path': current_dir,
        'train': os.path.join(current_dir, 'business_cards/train/images'),
        'val': os.path.join(current_dir, 'business_cards/valid/images'),
        'names': {
            0: 'front',
            1: 'back'
        },
        'nc': 2
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Initialize and train model
    model = YOLO('yolov8n.pt')
    model.train(data='dataset.yaml', epochs=100)
    
    # First export to TFLite
    print("Exporting to TFLite format...")
    model.export(format='tflite', imgsz=640)

if __name__ == "__main__":
    train_and_export()