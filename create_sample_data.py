import pandas as pd
import numpy as np
import os

# Ensure the folder exists
os.makedirs("data/processed", exist_ok=True)

# 100 samples, 6 features
df = pd.DataFrame(
    np.random.rand(100, 6),
    columns=['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
)
df['Activity_Type'] = np.random.choice(['Walking', 'Running', 'Sitting'], size=100)

# Save CSV
df.to_csv("data/processed/activity_data.csv", index=False)

print("Sample dataset created at data/processed/activity_data.csv")
