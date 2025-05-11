import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_scaled, y)

# Save the model and scaler
import joblib
joblib.dump(rf, 'rf_tour_winner_model.joblib')
joblib.dump(scaler, 'scaler_tour_winner.joblib')

