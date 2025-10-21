import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset
df = pd.read_csv("heart.csv")  # Make sure heart.csv is in your repo root

# Map categorical columns if needed
# (Assuming heart.csv uses numeric codes already; adjust if it uses strings)
# Example if you have 'sex' as 'Male'/'Female':
# df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})

# Define features and target
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal']
X = df[features]
y = df['target']

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
