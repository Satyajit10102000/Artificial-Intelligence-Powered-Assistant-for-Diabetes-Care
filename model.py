# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('diabetes.csv')

# Drop 'Pregnancies' column if exists
if 'Pregnancies' in data.columns:
    data = data.drop('Pregnancies', axis=1)

# Split features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))  # NOTE: No need of use_label_encoder
]

# Meta-model
meta_model = LogisticRegression()

# Stacking Classifier
stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=False,  # (Optional) to not pass original features
    n_jobs=-1  # (Optional) to use all CPU cores
)

# Train model
stacked_model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = stacked_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")

# Save the trained model and scaler
joblib.dump(stacked_model, 'diabetes_stacked_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("💾 Model and scaler saved successfully!")
