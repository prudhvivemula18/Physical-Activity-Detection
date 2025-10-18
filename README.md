# Physical Activity Detection using Sensor Data

This project classifies human physical activities (walking, standing, sitting, running, etc.) using motion-sensor data (accelerometer and gyroscope).  
It demonstrates an end-to-end AI/ML pipeline — from preprocessing and feature extraction to model training and deployment.

# Key Highlights
- Built complete ML pipeline for activity recognition.
- Baseline ML (Random Forest, XGBoost) and deep models (1D-CNN, LSTM).
- Evaluated on **UCI HAR Dataset**, achieving **XX % accuracy / YY % F1-score**.
 Deployed demo with **Streamlit** and **TensorFlow Lite**.

# Project Flow


# Repository Structure



# How to Run
```bash
git clone https://github.com/prudhvivemula18/Physical-Activity-Detection.git
cd Physical-Activity-Detection
pip install -r requirements.txt

python src/models/baseline_sklearn.py

streamlit run demos/streamlit_app.py
