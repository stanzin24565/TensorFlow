import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers, callbacks

# Load Custom Dataset
df = pd.read_csv("dataset.csv")  # Ensure this file exists
texts = df['text'].tolist()
labels = df['label'].tolist()

#Convert Labels to TensorFlow Format
labels = tf.convert_to_tensor(labels, dtype=tf.int32)

#Split Data (80% Training, 20% Testing)
split_index = int(0.8 * len(texts))
train_texts, test_texts = texts[:split_index], texts[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

#Tokenization & Vectorization
VOCAB_SIZE = 15000  # Increased vocabulary size
MAX_LEN = 300  # Increased max sequence length

vectorizer = TextVectorization(max_tokens=VOCAB_SIZE, output_mode="int", output_sequence_length=MAX_LEN)
vectorizer.adapt(train_texts)

#Convert to TensorFlow Dataset
train_data = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)).batch(64).prefetch(tf.data.AUTOTUNE)
test_data = tf.data.Dataset.from_tensor_slices((test_texts, test_labels)).batch(64).prefetch(tf.data.AUTOTUNE)

#Define Improved Model
model = tf.keras.Sequential([
    vectorizer,  # Convert text to tokenized sequences
    layers.Embedding(VOCAB_SIZE, 128, mask_zero=True),  # Increased embedding size
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),  # Bidirectional LSTM
    layers.Bidirectional(layers.LSTM(64)),  # Additional LSTM layer
    layers.Dropout(0.4),  # Dropout to prevent overfitting
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")  # Output layer for binary classification
])

#Define Optimizer & Learning Rate Schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

#Compile the Model
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

#Early Stopping (Stop Training if No Improvement)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#Train the Model
history = model.fit(train_data, validation_data=test_data, epochs=10, callbacks=[early_stopping])

#Evaluate Performance
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.4f}")

#Plot Accuracy Over Epochs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Save the Model
model.save("text_classification_model.h5")
print("Model saved successfully!")

#Test with New Sample
sample_review = ["The movie was absolutely amazing! Highly recommended."]
prediction = model.predict(sample_review)[0][0]
sentiment = "Positive " if prediction > 0.5 else "Negative "
print(f"Predicted Sentiment: {sentiment}")
