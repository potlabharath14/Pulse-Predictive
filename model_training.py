import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

warnings.filterwarnings('ignore')

print("--- 1. Loading Data ---")
df = pd.read_csv('patient_data.csv')
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ─── 2. Data Cleaning ───────────────────────────────────────────────────────
print("\n--- 2. Data Cleaning & Preparation ---")

# Rename columns for consistency
df.columns = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication', 'Severity',
              'BreathShortness', 'VisualChanges', 'NoseBleeding', 'WhenDiagnosed',
              'Systolic', 'Diastolic', 'ControlledDiet', 'Stages']

import re

# Strip whitespace from ALL string columns (including non-breaking spaces)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# Fix known typos and inconsistencies in dataset
df['Severity'] = df['Severity'].replace({'Sever': 'Severe'})
df['Stages'] = df['Stages'].replace({
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS',
    'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)'
})
# Normalize BP range formats (e.g. '121- 130' -> '121 - 130')
df['Systolic'] = df['Systolic'].str.replace(r'(\d+)-\s*(\d+)', r'\1 - \2', regex=True)
df['Diastolic'] = df['Diastolic'].str.replace(r'(\d+)-\s*(\d+)', r'\1 - \2', regex=True)

# Remove duplicates and NaN
duplicates = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"Removed {duplicates} duplicate rows.")

missing = df.isnull().sum().sum()
df.dropna(inplace=True)
print(f"Removed rows with missing values. Null count was {missing}.")
print(f"Cleaned shape: {df.shape}")

print(f"\nTarget distribution:")
print(df['Stages'].value_counts())
print(f"\nUnique values per column:")
for col in df.columns:
    print(f"  {col}: {df[col].unique()}")

# ─── 3. EDA ──────────────────────────────────────────────────────────────────
print("\n--- 3. Exploratory Data Analysis (EDA) ---")
eda_dir = 'eda_plots'
os.makedirs(eda_dir, exist_ok=True)

# 3.1 Target distribution
plt.figure(figsize=(8, 5))
colors_stages = {'HYPERTENSION (Stage-1)': '#3b82f6', 'HYPERTENSION (Stage-2)': '#f59e0b', 'HYPERTENSIVE CRISIS': '#ef4444'}
stage_counts = df['Stages'].value_counts()
plt.bar(stage_counts.index, stage_counts.values, color=[colors_stages.get(s, '#6366f1') for s in stage_counts.index])
plt.title('Hypertension Stage Distribution', fontsize=14)
plt.xlabel('Stage')
plt.ylabel('Count')
plt.xticks(rotation=15, fontsize=9)
plt.tight_layout()
plt.savefig(f'{eda_dir}/stage_distribution.png')
plt.close()

# 3.2 Gender vs Stages
plt.figure(figsize=(8, 5))
pd.crosstab(df['Gender'], df['Stages']).plot(kind='bar', color=['#3b82f6', '#f59e0b', '#ef4444'])
plt.title('Gender vs Hypertension Stages')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{eda_dir}/gender_vs_stages.png')
plt.close()

# 3.3 Age vs Stages
plt.figure(figsize=(8, 5))
pd.crosstab(df['Age'], df['Stages']).plot(kind='bar', color=['#3b82f6', '#f59e0b', '#ef4444'])
plt.title('Age Group vs Hypertension Stages')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{eda_dir}/age_vs_stages.png')
plt.close()

# 3.4 Severity pie
plt.figure(figsize=(6, 6))
df['Severity'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Severity Distribution')
plt.ylabel('')
plt.savefig(f'{eda_dir}/severity_pie.png')
plt.close()

print(f"EDA Complete! Plots saved into '{eda_dir}/' folder.")

# ─── 4. Encoding ─────────────────────────────────────────────────────────────
print("\n--- 4. Encoding & Feature Engineering ---")

# Separate target
X = df.drop('Stages', axis=1)
y = df['Stages']

# Encode all categorical features
label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"\n  Target (Stages): {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

# Correlation heatmap (after encoding)
plt.figure(figsize=(12, 9))
corr_df = X.copy()
corr_df['Stages'] = y_encoded
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'{eda_dir}/correlation_heatmap.png')
plt.close()

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ─── 5. Model Building ───────────────────────────────────────────────────────
print("\n--- 5. Model Building (Train-Test Split & Training) ---")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

results = []
best_acc = 0
best_model_name = ""

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    })

    if acc > best_acc:
        best_acc = acc
        best_model_name = name

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\nModel Comparison Table:")
print("-" * 65)
print(results_df.to_string(index=False))
print("-" * 65)

# ─── 6. Model Selection ──────────────────────────────────────────────────────
print(f"\n--- 6. Model Selection ---")
print(f"Best model: {best_model_name} with accuracy {best_acc:.4f}")
print("Using Logistic Regression for interpretability and SHAP compatibility.")

# Train final model
final_model = LogisticRegression(max_iter=1000, random_state=42)
final_model.fit(X_train, y_train)

y_pred_final = final_model.predict(X_test)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_final, target_names=target_encoder.classes_))

# ─── 7. Save Model ───────────────────────────────────────────────────────────
print("--- 7. Saving Model ---")
with open('logreg_model.pkl', 'wb') as file:
    pickle.dump({
        'model': final_model,
        'scaler': scaler,
        'encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_names': list(df.drop('Stages', axis=1).columns)
    }, file)

print("'logreg_model.pkl' created successfully! Ready for Flask deployment.")
