import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset
df = pd.read_csv("heart.csv")  # Make sure heart.csv is in the same folder

# Define features and target
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal']
X = df[features]
y = df['target']

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model as model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
