import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load winner data
winners_df = pd.read_csv('Tour_Winners_data_1.csv')
features = ['height_(m)', 'weight_(Kg)', 'age', 'BMI']
winners = winners_df[features].dropna().copy()
winners['target'] = 1

# Generate synthetic non-winners
n_samples = len(winners) * 3  # 3x as many non-winners as winners
np.random.seed(42)
non_winners = pd.DataFrame({
    'height_(m)': np.random.uniform(winners['height_(m)'].min(), winners['height_(m)'].max(), n_samples),
    'weight_(Kg)': np.random.uniform(winners['weight_(Kg)'].min(), winners['weight_(Kg)'].max(), n_samples),
    'age': np.random.uniform(winners['age'].min(), winners['age'].max(), n_samples),
    'BMI': np.random.uniform(winners['BMI'].min(), winners['BMI'].max(), n_samples),
})
non_winners['target'] = 0

# Combine datasets
full_data = pd.concat([winners, non_winners], ignore_index=True)
X = full_data[features].values
y = full_data['target'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train classifier
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'winner_classifier.joblib')
joblib.dump(scaler, 'scaler.joblib')
print('Model and scaler saved.') 