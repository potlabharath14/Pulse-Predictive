import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
import io
import re
import os
import logging
from datetime import datetime
from models.db import fs

matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML Artifacts
model = None
scaler = None
encoders = None
target_encoder = None
FEATURE_NAMES = []
explainer = None

STAGE_CONFIG = {
    'NORMAL': {'class': 'success', 'emoji': '🟢', 'label': 'Normal', 'color': '#22c55e'},
    'HYPERTENSION (Stage-1)': {'class': 'warning', 'emoji': '🟡', 'label': 'Stage 1 Hypertension', 'color': '#f59e0b'},
    'HYPERTENSION (Stage-2)': {'class': 'danger', 'emoji': '🔴', 'label': 'Stage 2 Hypertension', 'color': '#ef4444'},
    'HYPERTENSIVE CRISIS': {'class': 'crisis', 'emoji': '🚨', 'label': 'Hypertensive Crisis', 'color': '#dc2626'},
}

FEATURE_LABELS = ['Gender', 'Age Group', 'Family History', 'Patient Status', 'Medication',
                  'Severity', 'Breath Shortness', 'Visual Changes', 'Nose Bleeding',
                  'When Diagnosed', 'Systolic BP', 'Diastolic BP', 'Controlled Diet']

def load_ml_model():
    global model, scaler, encoders, target_encoder, FEATURE_NAMES, explainer
    try:
        with open('logreg_model.pkl', 'rb') as file:
            data = pickle.load(file)
            model = data['model']
            scaler = data['scaler']
            encoders = data['encoders']
            target_encoder = data['target_encoder']
            FEATURE_NAMES = data['feature_names']
            logger.info("ML Model loaded successfully.")
    except FileNotFoundError:
        logger.error("logreg_model.pkl not found! Please run 'python model_training.py' first.")
        return False
    
    try:
        df_raw = pd.read_csv('patient_data.csv')
        df_raw.columns = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication', 'Severity',
                        'BreathShortness', 'VisualChanges', 'NoseBleeding', 'WhenDiagnosed',
                        'Systolic', 'Diastolic', 'ControlledDiet', 'Stages']
        for col in df_raw.columns:
            if df_raw[col].dtype == object:
                df_raw[col] = df_raw[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        df_raw['Severity'] = df_raw['Severity'].replace({'Sever': 'Severe'})
        df_raw['Systolic'] = df_raw['Systolic'].str.replace(r'(\d+)-\s*(\d+)', r'\1 - \2', regex=True)
        df_raw['Diastolic'] = df_raw['Diastolic'].str.replace(r'(\d+)-\s*(\d+)', r'\1 - \2', regex=True)
        df_raw.drop_duplicates(inplace=True)
        df_raw.dropna(inplace=True)
        X_raw = df_raw.drop('Stages', axis=1)
        for col in X_raw.columns:
            X_raw[col] = encoders[col].transform(X_raw[col])
        X_train_scaled = scaler.transform(X_raw)
        explainer = shap.LinearExplainer(model, X_train_scaled)
        logger.info("SHAP explainer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading SHAP explainer: {str(e)}")
        return False
    
    return True

def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        for cls in encoder.classes_:
            if cls.strip() == value.strip():
                return encoder.transform([cls])[0]
        raise

def generate_shap_summary(shap_vals, feature_labels, predicted_class_idx):
    if len(shap_vals.shape) == 2:
        vals = shap_vals[:, predicted_class_idx]
    else:
        vals = shap_vals
    pairs = list(zip(feature_labels, vals))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = pairs[:3]
    parts = []
    for name, val in top:
        direction = "increased" if val > 0 else "decreased"
        parts.append(f"{name} {direction} risk (impact: {abs(val):.3f})")
    return "Main contributors: " + "; ".join(parts)

def predict_patient(form_data):
    if model is None:
        raise ValueError("Model not loaded. Call load_ml_model() first.")
        
    encoded_values = []
    for col in FEATURE_NAMES:
        encoded_values.append(safe_encode(encoders[col], form_data[col]))
        
    patient_df = pd.DataFrame([encoded_values], columns=FEATURE_NAMES)
    patient_scaled = scaler.transform(patient_df)
    
    prediction_idx = model.predict(patient_scaled)[0]
    probabilities = model.predict_proba(patient_scaled)[0]
    predicted_stage = target_encoder.inverse_transform([prediction_idx])[0]
    confidence = round(float(probabilities[prediction_idx]) * 100, 1)
    
    stage_info = STAGE_CONFIG.get(predicted_stage, STAGE_CONFIG['HYPERTENSION (Stage-1)'])
    
    prob_breakdown = []
    for i, cls in enumerate(target_encoder.classes_):
        cfg = STAGE_CONFIG.get(cls, {})
        prob_breakdown.append({
            'name': cfg.get('label', cls),
            'pct': round(float(probabilities[i]) * 100, 1),
            'class': cfg.get('class', 'warning')
        })
        
    shap_values = explainer.shap_values(patient_scaled)
    if isinstance(shap_values, list):
        sv_plot = shap_values[0][prediction_idx] if len(shap_values[0].shape) > 1 else shap_values[0]
        shap_summary = generate_shap_summary(shap_values[0], FEATURE_LABELS, prediction_idx)
    else:
        sv_plot = shap_values[0][:, prediction_idx] if len(shap_values.shape) == 3 else shap_values[0]
        shap_summary = generate_shap_summary(shap_values[0], FEATURE_LABELS, prediction_idx)
        
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#ef4444' if x > 0 else '#22c55e' for x in sv_plot]
    ax.barh(FEATURE_LABELS, sv_plot, color=colors, height=0.6, edgecolor='none')
    ax.set_facecolor('#0f0f1a')
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_xlabel("SHAP Value (Impact)", color='#94a3b8', fontsize=10)
    ax.set_title(f"Feature Impact for: {stage_info['label']}", color='white', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(colors='#e2e8f0', labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(x=0, color='#475569', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    
    plot_filename = f"shap_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.png"
    shap_buf = io.BytesIO()
    plt.savefig(shap_buf, dpi=150, facecolor='#0f0f1a', format='png')
    shap_buf.seek(0)
    shap_grid_id = fs.put(shap_buf.read(), filename=plot_filename, content_type='image/png')
    plt.close()
    
    os.makedirs('static/shap_plots', exist_ok=True)
    with open(os.path.join('static', 'shap_plots', plot_filename), 'wb') as f:
        f.write(fs.get(shap_grid_id).read())
        
    return {
        'predicted_stage': predicted_stage,
        'confidence': confidence,
        'stage_info': stage_info,
        'prob_breakdown': prob_breakdown,
        'shap_plot_path': plot_filename,
        'shap_summary': shap_summary,
        'shap_grid_id': shap_grid_id
    }
