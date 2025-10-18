# deep_lstm_cnn.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from src.data.load_ucihar import load_ucihar

# -------------------------------
# Step 1: Load UCI HAR dataset
# -------------------------------
X_train, y_train, X_test, y_test = load_ucihar()

# -------------------------------
# Step 2: Reshape for CNN/LSTM
# -------------------------------
X_train_dl = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_dl  = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# -------------------------------
# Step 3: Create CNN + LSTM model
# -------------------------------
def create_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_cnn_lstm_model(input_shape=(X_train_dl.shape[1], 1), num_classes=6)

# -------------------------------
# Step 4: Train the model
# -------------------------------
history = model.fit(
    X_train_dl, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# -------------------------------
# Step 5: Evaluate the model
# -------------------------------
loss, acc = model.evaluate(X_test_dl, y_test, verbose=0)
print(f"Deep Learning Model Accuracy: {acc:.4f}")

# -------------------------------
# Step 6: Save the trained model
# -------------------------------
os.makedirs("results", exist_ok=True)
model_path = "results/deep_lstm_cnn_model.h5"
model.save(model_path)
print(f"Trained deep learning model saved to {model_path}")
