from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define dataset paths
dataset_dir = r"C:\Users\HP\facial_health_recognition_system\dataset_preprocessed"

# Ensure dataset directory exists
if not os.path.exists(dataset_dir):
    print(f"❌ Error: Dataset directory not found: {dataset_dir}")
    exit(1)

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,         # Normalize pixel values
    rotation_range=30,         # Rotate images up to 30 degrees
    width_shift_range=0.2,     # Shift width by 20%
    height_shift_range=0.2,    # Shift height by 20%
    shear_range=0.2,           # Shear transformation
    zoom_range=0.2,            # Zoom in/out by 20%
    horizontal_flip=True,      # Flip images horizontally
    validation_split=0.2       # Split into train & validation sets (80-20)
)

# Train Generator (80% data)
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Training subset
)

# Validation Generator (20% data)
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Validation subset
)

# Print class indices (for debugging)
print(f"✅ Class Indices: {train_generator.class_indices}")
