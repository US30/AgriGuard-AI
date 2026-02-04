import os
import shutil
import random
from tqdm import tqdm

def organize_dataset():
    # 1. Define Paths
    # The Kaggle download usually unzips into this long nested path
    # Check your folder to be sure, but this is standard for 'vipoooool/new-plant-diseases-dataset'
    raw_base = "data/raw/images/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    
    # If the above path doesn't exist, try the non-augmented one or short path
    if not os.path.exists(raw_base):
        raw_base = "data/raw/images/New Plant Diseases Dataset(Augmented)"
        if not os.path.exists(raw_base):
            # Fallback: User might need to check manually where it unzipped
            print(f"[Error] Could not find dataset at: {raw_base}")
            print("Please check 'data/raw/images' to see the exact folder name.")
            return

    processed_dir = "data/processed/dataset_yolo"
    
    print(f"Source: {raw_base}")
    print(f"Destination: {processed_dir}")

    # 2. Setup Train/Val split
    split_ratio = 0.8 # 80% Train, 20% Val
    
    # Clean previous run
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    
    os.makedirs(os.path.join(processed_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, 'val'), exist_ok=True)

    # 3. Iterate over classes (folders)
    # The dataset has a 'train' and 'valid' folder already, but we'll merge and re-split 
    # to control the randomness and ensure we know exactly what's where.
    source_train = os.path.join(raw_base, 'train')
    
    classes = [d for d in os.listdir(source_train) if os.path.isdir(os.path.join(source_train, d))]
    print(f"Found {len(classes)} classes (Diseases/Crops).")

    for cls in tqdm(classes, desc="Processing Classes"):
        # Create class folders in dest
        os.makedirs(os.path.join(processed_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, 'val', cls), exist_ok=True)
        
        # Get all images for this class
        src_path = os.path.join(source_train, cls)
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle
        random.shuffle(images)
        
        # Split
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # Copy files
        for img in train_imgs:
            shutil.copy2(os.path.join(src_path, img), os.path.join(processed_dir, 'train', cls, img))
            
        for img in val_imgs:
            shutil.copy2(os.path.join(src_path, img), os.path.join(processed_dir, 'val', cls, img))

    print("\n[Success] Dataset organized for YOLOv8 Classification!")
    print(f"Location: {processed_dir}")

if __name__ == "__main__":
    organize_dataset()