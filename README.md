Physical Activity Detection
Overview

Physical Activity Detection is a machine learning and deep learning project that predicts the physical activity of a person (e.g., walking, sitting, standing, laying) based on smartphone sensor data. The system is built using Python and provides models for both classical machine learning (Random Forest) and deep learning (CNN+LSTM hybrid).

Technology Used

Python

Pandas

NumPy

Scikit-learn

TensorFlow / Keras

Matplotlib / Seaborn

Features

Data collection and loading from UCI HAR Dataset

Data preprocessing and feature extraction

Exploratory Data Analysis (EDA) with visualizations

Machine learning model building (Random Forest)

Deep learning model building (CNN+LSTM)

Model evaluation and metrics calculation

Saving trained models for real-time predictions

Ready for deployment or integration with smartphone-collected datasets

Data Collection

The dataset used is the UCI Human Activity Recognition (HAR) dataset, which contains accelerometer and gyroscope sensor data from smartphones worn by participants while performing six activities:

Walking

Walking Upstairs

Walking Downstairs

Sitting

Standing

Laying

Data Cleaning and Preprocessing

Loaded raw sensor readings from the UCI HAR dataset.

Removed null values and ensured consistent formatting.

Scaled and reshaped the data for compatibility with machine learning and deep learning models.

Split data into training and testing sets.

For deep learning, sequences were prepared for the CNN+LSTM model input.

Exploratory Data Analysis

Analyzed distribution of each activity class.

Visualized sensor readings using line plots and heatmaps.

Computed correlations between sensor axes and activities.

Generated statistics such as mean, standard deviation, and variance per activity.

Model Building and Selection

Random Forest Classifier: Classical ML approach achieving ~92.5% accuracy.

CNN+LSTM Hybrid Model: Captures both spatial and temporal features of the time-series data.

Compared models using precision, recall, F1-score, and overall accuracy.

Model Evaluation

Detailed classification report generated for both models.

Confusion matrices and other visualizations used to assess performance per activity class.

Trained models are saved as .pkl (Random Forest) and .h5 (CNN+LSTM) files for reuse.

Usage

To run the Physical Activity Detection project on your local machine:

Clone this repository:

git clone https://github.com/prudhvivemula18/Physical-Activity-Detection.git


Navigate to the project directory:

cd Physical-Activity-Detection


Create a virtual environment and activate it:

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac


Install dependencies:

pip install -r requirements.txt


Run the baseline model evaluation (Random Forest):

python src/models/baseline_sklearn.py


Run the CNN+LSTM model:

python src/models/deep_lstm_cnn.py


For using your own smartphone-recorded data, place it in the data/ folder and update the load_ucihar.py script to read your CSV files.

Results

Random Forest Model Accuracy: ~92.57%

CNN+LSTM Model: Captures temporal patterns in sensor data for robust activity detection.

Trained models are saved in the results/ folder.

Folder Structure
Physical-Activity-Detection/
├─ .venv/                  # Virtual environment
├─ data/                   # Raw and processed datasets
├─ demos/                  # Sample scripts or demos
├─ experiments/            # Experiment logs and intermediate files
├─ results/                # Saved trained models and outputs
├─ src/                    # Project source code
│  ├─ data/                # Data loading scripts
│  ├─ features/            # Feature extraction scripts
│  ├─ models/              # ML and DL models
│  ├─ utils/               # Helper scripts (visualization, evaluation)
├─ README.md               # Project description
├─ requirements.txt        # Python dependencies
└─ create_sample_data.py   # Script to create sample dataset

Skills & Tools

Programming & Libraries: Python, Pandas, NumPy, Scikit-learn, TensorFlow, Keras

Machine Learning: Random Forest, feature engineering, classification metrics

Deep Learning: CNN, LSTM, CNN+LSTM hybrid for sequence modeling

Data Handling & EDA: Data preprocessing, visualization, statistical analysis

Version Control & Collaboration: Git, GitHub