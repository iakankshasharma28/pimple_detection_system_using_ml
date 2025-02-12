import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# ✅ Step 1: Define Dataset Path
train_dir = "dataset_preprocessed/"  # Change to your dataset path

# ✅ Step 2: Data Preprocessing & Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize images (0 to 1)
    rotation_range=30,     # Rotate images randomly
    width_shift_range=0.2, # Shift images horizontally
    height_shift_range=0.2,# Shift images vertically
    shear_range=0.2,       # Shear transformation
    zoom_range=0.2,        # Zoom in/out
    horizontal_flip=True,  # Flip images
    validation_split=0.2   # 80% training, 20% validation split
)

# ✅ Training Data Generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# ✅ Validation Data Generator
validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ✅ Model Save Path
model_path = "pimple_detector_augmented.h5"

# ✅ Step 3: Define & Train Model (Only if Model is Not Already Saved)
if not os.path.exists(model_path):
    print("🚀 Training New Model...")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile Model
    model.compile(optimizer=Adam(learning_rate=0.0001),  # Reduced LR for stability
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Show Model Summary
    model.summary()

    # Train Model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=5,  # Increased epochs for better training
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator)
    )

    # Save Trained Model
    model.save(model_path)
    print(f"✅ Model saved as {model_path}")

    # ✅ Step 4: Plot Training Performance
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.show()

else:
    print("✅ Loading pre-trained model...")
    model = load_model(model_path)  # Load Saved Model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Step 5: Predict on a New Image
def predict_pimple(img_path, model):
    if not os.path.exists(img_path):
        print(f"⚠️ Image not found: {img_path}")
        return

    img = load_img(img_path, target_size=(224, 224))  # Load Image
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for model input

    prediction = model.predict(img_array)[0][0]
    label = "✅ No Pimples (Clear Skin) 😊" if prediction > 0.5 else "❌ Pimples Detected 😞"
    
    print(f"{img_path}: {label}")

# Example Usage
predict_pimple("dataset/test_image.jpg", model)

# ✅ Step 6: Predict Multiple Images in a Folder
def predict_multiple_images(folder_path, model):
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return

    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"⚠️ No images found in {folder_path}")
        return

    print(f"🔍 Found {len(image_files)} images in {folder_path}. Processing...\n")
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        # Load and preprocess the image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Reshape for model input

        # Get prediction
        prediction = model.predict(img_array)[0][0]
        label = "✅ No Pimples (Clear Skin) 😊" if prediction > 0.5 else "❌ Pimples Detected 😞"
        
        print(f"{img_file}: {label}")

# Example Usage for Multiple Images
test_folder = "dataset/test_images"  # Change this to your folder path
predict_multiple_images(test_folder, model)
