from ultralytics import YOLO
import os

def train_model():
    # 1. Define Path to Dataset
    # YOLO needs the absolute path usually, or relative to current dir
    dataset_path = os.path.abspath("data/processed/dataset_yolo")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Run 'src/02_data_preprocessing.py' first!")

    print(f"Starting Training on: {dataset_path}")
    
    # 2. Load Model
    # 'yolov8n-cls.pt' is the nano classification model (pre-trained on ImageNet)
    # It will transfer-learn to your crops automatically.
    model = YOLO('yolov8n-cls.pt') 

    # 3. Train
    # epochs=10 is enough to see results (Use 50 for final resume project)
    # imgsz=224 is standard for classification
    results = model.train(
        data=dataset_path, 
        epochs=5,           # Set to 20-50 for high accuracy if you have time
        imgsz=224, 
        batch=32,
        name='agriguard_model', # Results saved to runs/classify/agriguard_model
        project='weights'       # Save location
    )

    # 4. Validation (Optional, automatically done during train)
    print("\nTraining Complete. Validating...")
    metrics = model.val()
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")

    # 5. Export for Deployment
    # We export to ONNX format (standard for web deployment) or just keep .pt
    success = model.export(format='onnx')
    print(f"Model exported: {success}")

if __name__ == "__main__":
    train_model()