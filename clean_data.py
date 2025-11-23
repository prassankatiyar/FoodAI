import os
from PIL import Image

dataset_path = "dataset"

print(f"Checking images in '{dataset_path}'...")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                
                if img.mode in ('P', 'RGBA', 'LA'):
                    rgb_img = img.convert('RGB')
                    rgb_img.save(file_path)
                    print(f"Fixed format for: {file}")
                    
            except Exception as e:
                print(f"Error reading {file}: {e}")

print("Cleanup Complete! You can run training again.")