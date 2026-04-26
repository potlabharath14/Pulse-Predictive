import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

n_samples = 1000

# Features
gender = np.random.choice(['Male', 'Female'], n_samples)
age = np.random.randint(25, 80, n_samples)
family_history = np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6])
medication = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
symptoms = np.random.choice(['No Symptoms', 'Headache', 'Dizziness', 'Chest Pain', 'Shortness of Breath'], n_samples, p=[0.5, 0.2, 0.1, 0.1, 0.1])
lifestyle = np.random.choice(['Active', 'Moderate', 'Sedentary'], n_samples, p=[0.3, 0.4, 0.3])

# Generating target (Hypertension: 0=No, 1=Yes) based on logical medical conditions
systolic = []
diastolic = []
target = []

for i in range(n_samples):
    risk_score = 0
    if age[i] > 50: risk_score += 2
    if family_history[i] == 'Yes': risk_score += 2
    if lifestyle[i] == 'Sedentary': risk_score += 1
    if symptoms[i] in ['Chest Pain', 'Shortness of Breath']: risk_score += 2
    
    if risk_score >= 4 or (np.random.rand() > 0.85):
        sys = np.random.randint(130, 180)
        dia = np.random.randint(80, 110)
        t = 1
    elif risk_score >= 2:
        sys = np.random.randint(120, 140)
        dia = np.random.randint(75, 90)
        t = 1 if sys >= 130 or dia >= 80 else 0
    else:
        sys = np.random.randint(90, 120)
        dia = np.random.randint(60, 80)
        t = 0
        
    systolic.append(sys)
    diastolic.append(dia)
    target.append(t)

df = pd.DataFrame({
    'Gender': gender,
    'Age': age,
    'Family_History': family_history,
    'Medication': medication,
    'Symptoms': symptoms,
    'Lifestyle': lifestyle,
    'Systolic_BP': systolic,
    'Diastolic_BP': diastolic,
    'Hypertension': target
})

# Add some duplicates and missing values to demonstrate data cleaning in model_training.py
df_duplicates = df.head(10)
df = pd.concat([df, df_duplicates], ignore_index=True)
df.loc[15, 'Age'] = np.nan
df.loc[25, 'Systolic_BP'] = np.nan

df.to_csv('dataset.csv', index=False)
print("dataset.csv generated successfully with 1010 records (including anomalies).")
