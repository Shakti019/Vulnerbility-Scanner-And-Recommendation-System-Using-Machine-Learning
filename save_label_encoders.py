import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv('vulnerabilities.csv')

# Initialize the label encoders
severity_encoder = LabelEncoder()
affected_devices_encoder = LabelEncoder()

# Fit the label encoders on the data
severity_encoder.fit(df['Severity'])
affected_devices_encoder.fit(df['Affected Devices'])

# Save the label encoders to a pickle file
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'Severity': severity_encoder,
        'Affected Devices': affected_devices_encoder
    }, f)

print("Label encoders saved successfully!")
