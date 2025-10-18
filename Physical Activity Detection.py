import os
# from google.colab import drive
# drive.mount('/content/drive') 

import pandas as pd

train_df=pd.read_csv('/content/drive/My Drive/data/train.csv')

test_df=pd.read_csv('/content/drive/My Drive/data/test.csv')

train_df

test_df

train_df.columns

X_train=train_df.drop(['subject', 'Activity'],axis=1)
y_train=train_df["Activity"]

X_test=test_df.drop(['subject', 'Activity'],axis=1)
y_test=test_df["Activity"]

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# Assuming X_train is your training data (features)
X = X_train  # Assign X_train to X
y = y_train

# Step 3: Feature Scaling (Standardize the features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ... (rest of your code)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# Fitness function to evaluate classification accuracy
def fitness_function(selected_features, X, y):
    if not np.any(selected_features):  # Avoid all-zero selections
        return 0
    X_selected = X[:, selected_features == 1]
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    accuracy = cross_val_score(clf, X_selected, y, cv=3, scoring='accuracy').mean()
    return accuracy

# Simplified wolf update: all wolves copy the best one's features
def update_wolves(wolves, best_wolf):
    return np.tile(best_wolf, (wolves.shape[0], 1))

# Wolf Optimization Algorithm
def wolf_optimizer(X, y, num_features=20, num_wolves=5, max_iter=5):
    num_total_features = X.shape[1]
    wolves = np.random.randint(0, 2, size=(num_wolves, num_total_features))
    best_wolf = wolves[0].copy()
    best_fitness = 0

    for iteration in range(max_iter):
        # Evaluate all wolves in parallel
        fitness_scores = Parallel(n_jobs=-1)(delayed(fitness_function)(wolves[i], X, y) for i in range(num_wolves))

        best_index = np.argmax(fitness_scores)
        if fitness_scores[best_index] > best_fitness:
            best_fitness = fitness_scores[best_index]
            best_wolf = wolves[best_index].copy()

        wolves = update_wolves(wolves, best_wolf)

        print(f"Iteration {iteration + 1} - Best Fitness: {best_fitness:.4f}")

    selected_indices = np.where(best_wolf == 1)[0]
    return selected_indices[:num_features]

# === Main execution block ===

# Assume you already have X_scaled and y defined
# Replace this with your actual dataset if needed
# Example placeholder:
# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()
# X_scaled = data.data
# y = data.target
# import pandas as pd
# X = pd.DataFrame(X_scaled, columns=data.feature_names)

# Run optimizer
selected_features = wolf_optimizer(X_scaled, y, num_features=20)

print("Selected Feature Indices:", selected_features)

# Train classifier using selected features
X_selected = X_scaled[:, selected_features]
clf_final = LogisticRegression(max_iter=1000, solver='liblinear')
clf_final.fit(X_selected, y)

# Evaluate
y_pred = clf_final.predict(X_selected)
accuracy = accuracy_score(y, y_pred)
print("Training Accuracy with selected features:", accuracy)

# If original X is a DataFrame, get feature names
try:
    selected_feature_names = X.columns[selected_features]
    print("Selected Feature Names:", selected_feature_names.tolist())
except:
    print("Original feature names not available.")

# After running the optimizer and obtaining the selected features
Xtrain_selected = X.iloc[:, selected_features]

X_selected

y_train=y

Xtest_selected = X_test.iloc[:, selected_features]

Xtest_selected

y_test

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Encoding the target variable (Activity)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Optionally scale the features (using StandardScaler)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_selected)
X_test_scaled = scaler.transform(Xtest_selected)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


# Build the ANN model
model = Sequential()

# Input layer
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.3)) # Adding dropout

# Hidden layers
model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))


# Output layer (for multi-class classification)
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
 X_train_scaled, y_train_encoded,
 validation_split=0.2,
 epochs=100,
 batch_size=32,
 callbacks=[early_stop]
)

from sklearn.model_selection import train_test_split
# Manually split data into training and validation sets
X_train_split, X_val_scaled, y_train_split, y_val = train_test_split(
 X_train_scaled, y_train_encoded, test_size=0.2, random_state=42
)
history = model.fit(
 X_train_split, y_train_split,
 validation_data=(X_val_scaled, y_val), # Pass validation data explicitly
 epochs=100,
 batch_size=32,
 callbacks=[early_stop]
)

# Now you can correctly use X_val_scaled in predictions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# Get predictions for the validation set
y_val_pred_probs = model.predict(X_val_scaled) # Predict probabilities
y_val_pred = np.argmax(y_val_pred_probs, axis=1) # Convert to class labels

# Compute confusion matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Validation Set')
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_val, y_val_pred))
# Save the trained model
import joblib
model.save('classification_model.h5')
joblib.dump(label_encoder, 'label_encoder.pkl') # Save the encoder
joblib.dump(scaler, 'scaler.pkl') # Save the scale

# Plotting the training history
plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Save the trained model
model.save('classification_model.h5')

# pip install gradio

import gradio as gr
import numpy as np
import tensorflow as tf
import joblib

# Load the saved model, LabelEncoder, and StandardScaler
model = tf.keras.models.load_model('classification_model.h5')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

def predict_activity(*args):
    input_data = np.array(args).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction_probs = model.predict(input_scaled)
    prediction = np.argmax(prediction_probs, axis=1)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

inputs = [
    gr.Number(label='tBodyAcc-mean()-X'),
    gr.Number(label='tBodyAcc-mean()-Y'),
    gr.Number(label='tBodyAcc-mean()-Z'),
    gr.Number(label='tBodyAcc-std()-X'),
    gr.Number(label='tBodyAcc-std()-Z'),
    gr.Number(label='tBodyAcc-mad()-Y'),
    gr.Number(label='tBodyAcc-max()-X'),
    gr.Number(label='tBodyAcc-max()-Y'),
    gr.Number(label='tBodyAcc-energy()-X'),
    gr.Number(label='tBodyAcc-energy()-Z'),
    gr.Number(label='tBodyAcc-iqr()-Z'),
    gr.Number(label='tBodyAcc-entropy()-X'),
    gr.Number(label='tBodyAcc-entropy()-Y'),
    gr.Number(label='tBodyAcc-arCoeff()-X,1'),
    gr.Number(label='tBodyAcc-arCoeff()-X,3'),
    gr.Number(label='tBodyAcc-arCoeff()-Y,1'),
    gr.Number(label='tBodyAcc-arCoeff()-Y,2'),
    gr.Number(label='tBodyAcc-arCoeff()-Y,3'),
    gr.Number(label='tBodyAcc-arCoeff()-Y,4'),
    gr.Number(label='tBodyAcc-arCoeff()-Z,1')
]

interface = gr.Interface(
    fn=predict_activity,
    inputs=inputs,
    outputs="text",
    title="Human Activity Recognition",
    description="Enter the feature values and click 'Submit' to predict the activity class.",
    live=False,
)

interface.launch(share=True)

