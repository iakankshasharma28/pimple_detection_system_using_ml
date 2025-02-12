import cv2
import os
import numpy as np

# Define dataset paths
dataset_dir = r"C:\Users\HP\facial_health_recognition_system\dataset"
output_dir = r"C:\Users\HP\facial_health_recognition_system\dataset_preprocessed"

# Ensure dataset directory exists
if not os.path.exists(dataset_dir):
    print(f"❌ Error: Dataset directory not found: {dataset_dir}")
    exit(1)

def preprocess_images(input_folder, output_folder, size=(224, 224)):
    """Preprocess images by resizing and normalizing."""
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"⚠️ Warning: Folder not found: {input_folder}")
        return  # Skip this folder

    image_files = os.listdir(input_folder)
    
    if not image_files:
        print(f"⚠️ Warning: No images found in {input_folder}")
        return  # Skip empty folders

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        
        # Read image
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Error: Unable to read {img_path}, skipping.")
            continue  # Skip unreadable files

        # Resize and normalize
        img = cv2.resize(img, size)
        img = img / 255.0  # Normalize pixel values

        # Save preprocessed image (convert back to 0-255 before saving)
        cv2.imwrite(os.path.join(output_folder, img_name), img * 255)

# Process both categories
categories = ["pimple", "clear_skin"]  # ✅ Using correct folder names
for category in categories:
    input_folder = os.path.join(dataset_dir, category)
    output_folder = os.path.join(output_dir, category)
    
    preprocess_images(input_folder, output_folder)

print(f"✅ Preprocessing completed! Preprocessed images saved in: {output_dir}")
