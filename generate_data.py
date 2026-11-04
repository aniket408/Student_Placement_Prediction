
import pandas as pd
import numpy as np
from pathlib import Path

# Generate 1000 student records
np.random.seed(42)

number_of_students = 1000

data = {
    "CGPA": np.round(np.random.uniform(4.0, 10.0, number_of_students), 2),
    "10th_Percentage": np.round(np.random.uniform(40, 100, number_of_students), 2),
    "12th_Percentage": np.round(np.random.uniform(40, 100, number_of_students), 2),
    "Internships": np.random.randint(0, 4, number_of_students),
    "Projects": np.random.randint(0, 6, number_of_students),
    "Aptitude_Score": np.random.randint(30, 101, number_of_students),
    "Communication_Skills": np.random.randint(1, 11, number_of_students),
}

df = pd.DataFrame(data)

# Calculate placement score
placement_score = (
    df["CGPA"] * 8
    + df["10th_Percentage"] * 0.15
    + df["12th_Percentage"] * 0.20
    + df["Internships"] * 5
    + df["Projects"] * 2
    + df["Aptitude_Score"] * 0.20
    + df["Communication_Skills"] * 1.5
)

# Create placement status
threshold = placement_score.median()

df["Placement_Status"] = np.where(
    placement_score >= threshold,
    1,
    0
)

# Save CSV file
project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

data_folder.mkdir(exist_ok=True)

file_path = data_folder / "placement.csv"

df.to_csv(file_path, index=False)

print("Dataset generated successfully!")
print("Total Students:", len(df))
print("Dataset Shape:", df.shape)

print("\nPlacement Distribution:")
print(df["Placement_Status"].value_counts())

print("\nFirst 5 Records:")
print(df.head())

print("\nSaved Location:", file_path)
