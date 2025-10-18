# Physical Activity Detection

## Overview
**Physical Activity Detection** is a machine learning project that identifies human activities based on sensor data collected from smartphones. The system classifies activities such as walking, sitting, standing, and more using both classical machine learning and deep learning models.

---

## Technologies & Tools Used
- **Programming Language:** Python  
- **Libraries:** Scikit-learn, TensorFlow, Keras, Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Optional for future extension  
- **Dataset:** UCI Human Activity Recognition (HAR) Using Smartphones

---

## Features
- **Data Loading:** Efficiently loads UCI HAR dataset or user-collected sensor data.  
- **Data Preprocessing:** Cleans and structures the data for modeling.  
- **Model Building:**  
  - Classical Machine Learning: Random Forest Classifier  
  - Deep Learning: CNN + LSTM hybrid model  
- **Model Evaluation:** Accuracy, precision, recall, F1-score with detailed classification reports.  
- **Model Saving:** Trained models can be saved and reused without retraining.  

---

## Data Collection
- UCI HAR Dataset was used for training and testing.  
- Can also accept **real-time data recorded from a smartphone** using accelerometer and gyroscope sensors.

---

## Data Preprocessing
- Handles missing values and duplicates.  
- Normalizes features for optimal model performance.  
- Supports conversion of raw sensor readings into sequences suitable for LSTM input.

---

## Model Building & Evaluation
- Multiple models tested to select the best performer.  
- **Random Forest Classifier:** Achieved high accuracy (~92%) on test data.  
- **CNN + LSTM:** Captures temporal dependencies in sensor sequences for better activity recognition.  
- Evaluations include **accuracy, precision, recall, F1-score**.  
- Trained models are saved for easy reuse.

---

## Results
- **Random Forest Model Accuracy:** 0.9257  
- **CNN + LSTM Model:** High performance for sequential data  
- Models saved in `results/` folder for inference.

---

## Usage
1. Clone the repository:  
   ```bash
   git clone https://github.com/prudhvivemula18/Physical-Activity-Detection.git


Install required packages:

pip install -r requirements.txt


Run Python scripts to train or test models:
 python src/models/baseline_sklearn.py
 python src/models/deep_lstm_cnn.py


Skills Gained

Python programming and machine learning pipeline development

Data preprocessing and feature engineering

Model evaluation and performance metrics

Deep learning with CNN and LSTM

Version control using Git and GitHub

Folder Structure
Physical-Activity-Detection/
│
├── src/                    # Source code for models, data loading, and evaluation
├── data/                   # Dataset folder
├── results/                # Trained model outputs
├── create_sample_data.py   # Script to create sample dataset
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation


Note: You can replace the UCI HAR dataset with real-time smartphone sensor data by following the same preprocessing steps.