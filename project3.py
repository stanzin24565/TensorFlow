# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# 1. Dataset Preparation
# Assuming you've downloaded a landmark dataset from Kaggle (e.g., Google Landmarks Dataset)
# Let's create a function to load and preprocess the data

def load_dataset(data_dir, sample_size=0.1):
    """
    Load images and landmarks from dataset directory
    """
    # In a real scenario, you would load the CSV files that come with the dataset
    # For this example, we'll simulate loading image paths and landmarks
    
    # Get all image paths
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    # Sample a portion of the data if needed (for faster prototyping)
    if sample_size < 1.0:
        image_paths = np.random.choice(image_paths, size=int(len(image_paths)*sample_size), replace=False)
    
    # Create dummy landmarks (in a real project, you would load these from annotations)
    # Each landmark is represented as (x1, y1, x2, y2, ..., xn, yn) for n landmarks
    num_landmarks = 5  # Example: detecting 5 key points
    landmarks = np.random.rand(len(image_paths), num_landmarks * 2) * 0.8 + 0.1  # Random normalized coordinates
    
    return image_paths, landmarks

# 2. Data Loading and Preprocessing
def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0, 1]
    return img

def data_generator(image_paths, landmarks, batch_size=32, target_size=(224, 224)):
    """Generator function to yield batches of data"""
    num_samples = len(image_paths)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    while True:
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_images = []
            batch_landmarks = []
            
            for idx in batch_indices:
                img = preprocess_image(image_paths[idx], target_size)
                batch_images.append(img)
                batch_landmarks.append(landmarks[idx])
            
            yield np.array(batch_images), np.array(batch_landmarks)

# 3. Model Architecture
def create_landmark_model(input_shape=(224, 224, 3), num_landmarks=10):
    """Create a CNN model for landmark detection"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_landmarks * 2)(x)  # 2 coordinates per landmark
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

# 4. Training Setup
def train_model(data_dir, epochs=20, batch_size=32):
    # Load dataset
    image_paths, landmarks = load_dataset(data_dir, sample_size=0.5)  # Using 50% of data for speed
    
    # Split into train and validation sets
    train_paths, val_paths, train_landmarks, val_landmarks = train_test_split(
        image_paths, landmarks, test_size=0.2, random_state=42
    )
    
    # Create data generators
    train_gen = data_generator(train_paths, train_landmarks, batch_size)
    val_gen = data_generator(val_paths, val_landmarks, batch_size)
    
    # Calculate steps per epoch
    train_steps = len(train_paths) // batch_size
    val_steps = len(val_paths) // batch_size
    
    # Create model
    num_landmarks = train_landmarks.shape[1] // 2  # Each landmark has x and y coordinates
    model = create_landmark_model(num_landmarks=num_landmarks)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('best_landmark_model.h5', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history

# 5. Evaluation and Visualization
def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, image_paths, landmarks, num_samples=5):
    """Visualize model predictions on sample images"""
    plt.figure(figsize=(15, 10))
    
    for i in range(num_samples):
        idx = np.random.randint(len(image_paths))
        img = preprocess_image(image_paths[idx])
        true_landmarks = landmarks[idx].reshape(-1, 2)
        
        # Make prediction
        pred_landmarks = model.predict(np.expand_dims(img, axis=0))[0].reshape(-1, 2)
        
        # Plot image with landmarks
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        
        # Plot true landmarks (green)
        plt.scatter(true_landmarks[:, 0] * img.shape[1], 
                   true_landmarks[:, 1] * img.shape[0], 
                   c='g', s=50, label='True')
        
        # Plot predicted landmarks (red)
        plt.scatter(pred_landmarks[:, 0] * img.shape[1], 
                   pred_landmarks[:, 1] * img.shape[0], 
                   c='r', s=50, label='Predicted')
        
        plt.title(f'Sample {i+1}')
        plt.axis('off')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set the dataset directory (replace with your actual path)
    DATA_DIR = "path/to/your/kaggle/dataset"
    
    # Train the model
    print("Starting training...")
    model, history = train_model(DATA_DIR, epochs=30, batch_size=32)
    
    # Evaluate and visualize
    print("Training complete. Evaluating...")
    plot_training_history(history)
    
    # Load some data for visualization
    image_paths, landmarks = load_dataset(DATA_DIR, sample_size=0.1)
    visualize_predictions(model, image_paths, landmarks)
    
    # Save the final model
    model.save('landmark_detection_model.h5')
    print("Model saved as 'landmark_detection_model.h5'")