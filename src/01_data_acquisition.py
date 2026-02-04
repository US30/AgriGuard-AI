import os
import shutil
from dotenv import load_dotenv

# 1. Load Credentials
load_dotenv()

# Validate keys
if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
    raise ValueError(
        "Error: Kaggle credentials not found in .env file.\n"
        "Please ensure you have run 'setup_security.py' and pasted your keys into .env"
    )

from kaggle.api.kaggle_api_extended import KaggleApi

def download_real_agri_data():
    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()

    # Define target directories based on the setup script
    images_dir = "data/raw/images"
    finance_dir = "data/raw/financial"
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(finance_dir, exist_ok=True)

    print("Starting Real Data Download...")

    # --- DATASET 1: IMAGES (The "Vision" Part) ---
    # Source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
    # This is the standard "PlantVillage" expanded dataset (87K images)
    print("\n1. Downloading Crop Disease Images (This is large ~2GB, please wait)...")
    try:
        api.dataset_download_files(
            'vipoooool/new-plant-diseases-dataset', 
            path=images_dir, 
            unzip=True
        )
        print("   [Success] Image dataset downloaded.")
        
        # Cleanup: The dataset often unzips into nested folders. Let's organize it.
        # Structure usually looks like: data/raw/images/New Plant Diseases Dataset(Augmented)/...
        # We will deal with pathing in Phase 2.
        
    except Exception as e:
        print(f"   [Error] Could not download Image Data: {e}")

    # --- DATASET 2: FINANCIAL/YIELD CONTEXT (The "Fintech" Part) ---
    # Source: https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset
    # Contains: State, Crop, Season, Yield, Production, Rainfall
    print("\n2. Downloading Indian Crop Yield & Production Data...")
    try:
        api.dataset_download_files(
            'akshatgupta7/crop-yield-in-indian-states-dataset', 
            path=finance_dir, 
            unzip=True
        )
        print("   [Success] Financial/Yield dataset downloaded.")
    except Exception as e:
        print(f"   [Error] Could not download Yield Data: {e}")

    # --- DATASET 3: MARKET PRICES (Optional Context) ---
    # We try to get Daily Market Prices (Mandi Data) if available
    # Source: https://www.kaggle.com/datasets/kianwee/agricultural-raw-material-prices-19902020
    print("\n3. Downloading Agricultural Market Prices...")
    try:
        api.dataset_download_files(
            'kianwee/agricultural-raw-material-prices-19902020', 
            path=finance_dir, 
            unzip=True
        )
        print("   [Success] Market Price dataset downloaded.")
    except Exception as e:
        print("   [Warning] Could not download Price Data (We can proceed with Yield data only).")

    print("\n" + "="*40)
    print("PHASE 1 COMPLETE: RAW DATA ACQUIRED")
    print("="*40)

def inspect_downloads():
    print("\nVerifying Data Integrity:")
    
    # Check Images
    img_path = "data/raw/images"
    if os.path.exists(img_path):
        contents = os.listdir(img_path)
        print(f" - Image Folder contains: {len(contents)} items (Folders/Files)")
        if len(contents) > 0:
            print(f"   Sample: {contents[:3]}")
    else:
        print(" - Image Folder is empty!")

    # Check Finance
    fin_path = "data/raw/financial"
    if os.path.exists(fin_path):
        files = [f for f in os.listdir(fin_path) if f.endswith('.csv')]
        print(f" - Financial Folder contains: {files}")
    else:
        print(" - Financial Folder is empty!")

if __name__ == "__main__":
    download_real_agri_data()
    inspect_downloads()