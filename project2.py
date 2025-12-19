# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# 1.Dataset Preparation
# (Assuming dataset is structured in folders by class)
DATA_DIR = "path/to/your/kaggle/pet_faces_dataset"  
CLASS_NAMES = os.listdir(DATA_DIR)  
NUM_CLASSES = len(CLASS_NAMES)

# 2.Data Loading & Preprocessing
# Using ImageDataGenerator for augmentation & normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,          
    rotation_range=20,       
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,      
    zoom_range=0.2,        
    horizontal_flip=True,  
    validation_split=0.2     
)

# Load and augment training data
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(150, 150),  # Resize images
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load validation data (no augmentation)
val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 3.Model Architecture (CNN)
model = Sequential([
    # Convolutional Layers
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Fully Connected Layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(NUM_CLASSES, activation='softmax')  # Output layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4.Training the Model
# Callbacks for early stopping & saving best model
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_pet_classifier.h5', save_best_only=True)
]

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks
)

# 5. Evaluation & Visualization
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the final model
model.save('pet_face_classifier.h5')
print("Model saved as 'pet_face_classifier.h5'")