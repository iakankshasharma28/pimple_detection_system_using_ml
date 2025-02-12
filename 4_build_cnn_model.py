import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# Step 1: Define Paths
train_dir = "dataset_preprocessed/"  # Change to your dataset path
test_image_path = "dataset/test_image.jpg"  # Path to the test image

# Step 2: Data Preprocessing & Augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize images (0 to 1)
    rotation_range=30,  # Rotate images randomly
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2,  # Shift images vertically
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True,  # Flip images
    validation_split=0.2  # 80% training, 20% validation split
)

# Training Data Generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation Data Generator
validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Step 3: Define the CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.6),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001),  # Reduced LR for stability
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Show Model Summary
model.summary()

# Step 4: Train the Model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # Increased epochs for better training
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator)
)

# Step 5: Evaluate the Model
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')

plt.show()

# Step 6: Test the Model on a New Image from the dataset folder
def predict_pimple(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        print("✅ No Pimples (Clear Skin) 😊")
    else:
        print("❌ Pimples Detected 😞")

# Example usage with new image path
predict_pimple(test_image_path, model)

# Step 7: Save and Load Model
model.save("pimple_detector_augmented.h5")

# To reload the model later:
# from tensorflow.keras.models import load_model
# model = load_model("pimple_detector_augmented.h5")
