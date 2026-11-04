import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "placement.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "placement_model.joblib"

# Load dataset
df = pd.read_csv(DATA_PATH)

# input feature
x = df.drop("Placement_Status", axis=1)

# Target column
y = df["Placement_Status"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# create ml model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

# Train model
model.fit(x_train, y_train)

# predictions
y_pred = model.predict(x_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Model Path:", MODEL_PATH)