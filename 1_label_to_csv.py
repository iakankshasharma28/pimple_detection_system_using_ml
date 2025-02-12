import os
import pandas as pd

# Define the dataset directory
dataset_dir = r"C:\Users\HP\facial_health_recognition_system\dataset"

# Ensure dataset directory exists
if not os.path.exists(dataset_dir):
    print(f"❌ Error: Dataset directory not found: {dataset_dir}")
    exit(1)  # Stop execution if the main directory is missing

data = []
categories = ["pimple", "clear_skin"]  # ✅ Updated category names to match your folders

for category in categories:
    folder_path = os.path.join(dataset_dir, category)

    # Check if category folder exists
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder not found: {folder_path}")
        continue  # Skip missing folders instead of crashing

    # Get image files (Skip if the folder is empty)
    image_files = os.listdir(folder_path)
    if not image_files:
        print(f"⚠️ Warning: No images found in {folder_path}")
        continue  

    for img_name in image_files:
        data.append([img_name, category])

# Create DataFrame
df = pd.DataFrame(data, columns=["filename", "label"])

# Define output CSV path
output_csv = os.path.join(dataset_dir, "dataset_labels.csv")

# Save to CSV
df.to_csv(output_csv, index=False)

print(f"✅ CSV file saved successfully at: {output_csv}")

