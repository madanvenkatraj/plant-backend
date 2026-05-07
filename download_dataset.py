import os
import sys

def setup_kaggle():
    print("=============================================")
    print("   Plant AI - Dataset Downloader (Kaggle)    ")
    print("=============================================")
    
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json_path):
        print("To download the dataset, we need your Kaggle API credentials.")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll down to 'API' and click 'Create New Token'")
        print("3. Open the downloaded kaggle.json file")
        print("\nPlease enter the details from your kaggle.json file below:")
        username = input("Enter your Kaggle username: ").strip()
        key = input("Enter your Kaggle key: ").strip()
        
        with open(kaggle_json_path, 'w') as f:
            f.write(f'{{"username":"{username}","key":"{key}"}}')
            
        # Set permissions for the kaggle.json file (Linux/Mac)
        if os.name != 'nt':
            os.chmod(kaggle_json_path, 0o600)
            
        print("[OK] Credentials saved successfully!")
    else:
        print("[OK] Kaggle credentials found.")

    try:
        import kaggle
    except ImportError:
        print("\nInstalling Kaggle library...")
        os.system(f"{sys.executable} -m pip install kaggle")
        import kaggle

    dataset_name = "emmarex/plantdisease"
    download_dir = os.path.join(os.path.dirname(__file__), "dataset")
    
    print(f"\nDownloading dataset '{dataset_name}'...")
    print("This might take a while (approx. 800MB)...")
    
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_name, path=download_dir, unzip=True)
    
    print("\n[OK] Dataset downloaded and extracted successfully!")
    print("\nYou can now run the training script:")
    print(f"python train.py")

if __name__ == "__main__":
    setup_kaggle()
