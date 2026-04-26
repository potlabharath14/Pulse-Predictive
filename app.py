from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from bson.objectid import ObjectId
import pickle
import numpy as np
import os
import io
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd
from fpdf import FPDF
from pymongo import MongoClient
import gridfs

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey_change_in_production'

# ─── MongoDB Connection ─────────────────────────────────────────────────────
client = MongoClient('mongodb://localhost:27017/mlde')
mongo_db = client['mlde']
users_col = mongo_db['users']
predictions_col = mongo_db['predictions']
reports_col = mongo_db['reports']
fs = gridfs.GridFS(mongo_db)

users_col.create_index('username', unique=True)
predictions_col.create_index('user_id')
reports_col.create_index('prediction_id')

# ─── Flask-Login Setup ──────────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class MongoUser(UserMixin):
    def __init__(self, user_doc):
        self.user_doc = user_doc
        self.id = str(user_doc['_id'])
        self.username = user_doc['username']

@login_manager.user_loader
def load_user(user_id):
    try:
        doc = users_col.find_one({'_id': ObjectId(user_id)})
        if doc:
            return MongoUser(doc)
    except Exception:
        pass
    return None

# ─── Load ML Artifacts ──────────────────────────────────────────────────────
try:
    with open('logreg_model.pkl', 'rb') as file:
        data = pickle.load(file)
        model = data['model']
        scaler = data['scaler']
        encoders = data['encoders']
        target_encoder = data['target_encoder']
        FEATURE_NAMES = data['feature_names']
except FileNotFoundError:
    print("WARNING: logreg_model.pkl not found! Please run 'python model_training.py' first.")
    exit(1)

os.makedirs('static/shap_plots', exist_ok=True)
os.makedirs('static/reports', exist_ok=True)

# Prepare SHAP explainer
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

FEATURE_LABELS = ['Gender', 'Age Group', 'Family History', 'Patient Status', 'Medication',
                  'Severity', 'Breath Shortness', 'Visual Changes', 'Nose Bleeding',
                  'When Diagnosed', 'Systolic BP', 'Diastolic BP', 'Controlled Diet']

# Stage display config
STAGE_CONFIG = {
    'NORMAL': {'class': 'success', 'emoji': '🟢', 'label': 'Normal', 'color': '#22c55e'},
    'HYPERTENSION (Stage-1)': {'class': 'warning', 'emoji': '🟡', 'label': 'Stage 1 Hypertension', 'color': '#f59e0b'},
    'HYPERTENSION (Stage-2)': {'class': 'danger', 'emoji': '🔴', 'label': 'Stage 2 Hypertension', 'color': '#ef4444'},
    'HYPERTENSIVE CRISIS': {'class': 'crisis', 'emoji': '🚨', 'label': 'Hypertensive Crisis', 'color': '#dc2626'},
}


def safe_encode(encoder, value):
    """Safely encode a value, handling minor whitespace differences in the training data."""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Try with/without trailing space
        for cls in encoder.classes_:
            if cls.strip() == value.strip():
                return encoder.transform([cls])[0]
        raise


def generate_shap_summary(shap_vals, feature_labels, predicted_class_idx):
    """Return human-readable text for the predicted class."""
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


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user_doc = users_col.find_one({'username': username})
        if user_doc and check_password_hash(user_doc['password'], password):
            login_user(MongoUser(user_doc))
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your username and password.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if users_col.find_one({'username': username}):
            flash('Username already exists.')
            return redirect(url_for('register'))
        hashed = generate_password_hash(password)
        result = users_col.insert_one({'username': username, 'password': hashed, 'created_at': datetime.utcnow()})
        user_doc = users_col.find_one({'_id': result.inserted_id})
        login_user(MongoUser(user_doc))
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    history = list(predictions_col.find({'user_id': current_user.id}).sort('date', -1))
    return render_template('dashboard.html', history=history)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Capture form inputs
            form_data = {
                'Gender': request.form['gender'],
                'Age': request.form['age'],
                'History': request.form['history'],
                'Patient': request.form['patient'],
                'TakeMedication': request.form['take_medication'],
                'Severity': request.form['severity'],
                'BreathShortness': request.form['breath_shortness'],
                'VisualChanges': request.form['visual_changes'],
                'NoseBleeding': request.form['nose_bleeding'],
                'WhenDiagnosed': request.form['when_diagnosed'],
                'Systolic': request.form['systolic'],
                'Diastolic': request.form['diastolic'],
                'ControlledDiet': request.form['controlled_diet'],
            }

            # Encode each feature
            encoded_values = []
            for col in FEATURE_NAMES:
                encoded_values.append(safe_encode(encoders[col], form_data[col]))

            patient_df = pd.DataFrame([encoded_values], columns=FEATURE_NAMES)
            patient_scaled = scaler.transform(patient_df)

            # Predict
            prediction_idx = model.predict(patient_scaled)[0]
            probabilities = model.predict_proba(patient_scaled)[0]
            predicted_stage = target_encoder.inverse_transform([prediction_idx])[0]
            confidence = round(float(probabilities[prediction_idx]) * 100, 1)

            stage_info = STAGE_CONFIG.get(predicted_stage, STAGE_CONFIG['HYPERTENSION (Stage-1)'])

            # Build probability breakdown for all classes
            prob_breakdown = []
            for i, cls in enumerate(target_encoder.classes_):
                cfg = STAGE_CONFIG.get(cls, {})
                prob_breakdown.append({
                    'name': cfg.get('label', cls),
                    'pct': round(float(probabilities[i]) * 100, 1),
                    'class': cfg.get('class', 'warning')
                })

            # SHAP
            shap_values = explainer.shap_values(patient_scaled)
            shap_summary = generate_shap_summary(shap_values[0] if isinstance(shap_values, list) else shap_values, FEATURE_LABELS, prediction_idx)

            # For multi-class, shap_values may be a list of arrays
            if isinstance(shap_values, list):
                sv = shap_values[0][prediction_idx] if len(shap_values[0].shape) > 1 else shap_values[0]
            else:
                sv = shap_values[0][:, prediction_idx] if len(shap_values.shape) == 3 else shap_values[0]

            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ['#ef4444' if x > 0 else '#22c55e' for x in sv]
            ax.barh(FEATURE_LABELS, sv, color=colors, height=0.6, edgecolor='none')
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
            plot_path = os.path.join('static', 'shap_plots', plot_filename)
            plt.savefig(plot_path, dpi=150, facecolor='#0f0f1a')

            shap_buf = io.BytesIO()
            plt.savefig(shap_buf, dpi=150, facecolor='#0f0f1a', format='png')
            shap_buf.seek(0)
            shap_grid_id = fs.put(shap_buf.read(), filename=plot_filename, content_type='image/png')
            plt.close()

            # Save to MongoDB
            history_record_id = None
            if current_user.is_authenticated:
                doc = {
                    'user_id': current_user.id,
                    'date': datetime.utcnow(),
                    **form_data,
                    'predicted_stage': predicted_stage,
                    'confidence': confidence,
                    'risk_class': stage_info['class'],
                    'risk_label': stage_info['label'],
                    'shap_plot_path': plot_filename,
                    'shap_summary': shap_summary,
                    'shap_grid_id': shap_grid_id
                }
                ins = predictions_col.insert_one(doc)
                history_record_id = str(ins.inserted_id)

            return render_template('index.html',
                                   predicted_stage=predicted_stage,
                                   stage_info=stage_info,
                                   confidence=confidence,
                                   prob_breakdown=prob_breakdown,
                                   shap_plot=plot_filename,
                                   shap_summary=shap_summary,
                                   history_id=history_record_id)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template('index.html', error=f"A processing error occurred: {str(e)}")

    return render_template('index.html')


@app.route('/delete_prediction/<pred_id>')
@login_required
def delete_prediction(pred_id):
    try:
        predictions_col.delete_one({'_id': ObjectId(pred_id), 'user_id': current_user.id})
    except Exception:
        pass
    return redirect(url_for('dashboard'))


@app.route('/api/chart_data')
@login_required
def chart_data():
    history = list(predictions_col.find({'user_id': current_user.id}).sort('date', 1))
    labels = [h['date'].strftime('%d %b') for h in history]
    confidence = [h.get('confidence', 0) for h in history]

    stage_counts = {}
    for h in history:
        stage = h.get('risk_label', 'Unknown')
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    return jsonify({
        'labels': labels,
        'confidence': confidence,
        'distribution': stage_counts
    })


@app.route('/download_pdf/<pred_id>')
@login_required
def download_pdf(pred_id):
    history = predictions_col.find_one({'_id': ObjectId(pred_id), 'user_id': current_user.id})
    if not history:
        return "Not found", 404

    existing = reports_col.find_one({'prediction_id': pred_id})
    if existing and existing.get('pdf_grid_id'):
        pdf_file = fs.get(existing['pdf_grid_id'])
        return Response(pdf_file.read(), mimetype='application/pdf',
                        headers={'Content-Disposition': f"attachment; filename=Report_{history['date'].strftime('%Y%m%d')}.pdf"})

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, txt="Hypertension Predictive Report", ln=1, align="C")
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, txt="Generated by Pulse Predictive System", ln=1, align="C")
    pdf.ln(10)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Patient Information", ln=1)
    pdf.set_draw_color(100, 100, 241)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(95, 10, txt=f"Date: {history['date'].strftime('%Y-%m-%d %H:%M')}", ln=0)
    pdf.cell(95, 10, txt=f"Patient: {current_user.username}", ln=1)
    pdf.cell(95, 10, txt=f"Gender: {history.get('Gender', '')}", ln=0)
    pdf.cell(95, 10, txt=f"Age Group: {history.get('Age', '')}", ln=1)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Clinical Data", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(95, 10, txt=f"Systolic BP: {history.get('Systolic', '')}", ln=0)
    pdf.cell(95, 10, txt=f"Diastolic BP: {history.get('Diastolic', '')}", ln=1)
    pdf.cell(95, 10, txt=f"Severity: {history.get('Severity', '')}", ln=0)
    pdf.cell(95, 10, txt=f"Family History: {history.get('History', '')}", ln=1)
    pdf.cell(95, 10, txt=f"Medication: {history.get('TakeMedication', '')}", ln=0)
    pdf.cell(95, 10, txt=f"Controlled Diet: {history.get('ControlledDiet', '')}", ln=1)
    pdf.cell(95, 10, txt=f"Breath Shortness: {history.get('BreathShortness', '')}", ln=0)
    pdf.cell(95, 10, txt=f"Visual Changes: {history.get('VisualChanges', '')}", ln=1)
    pdf.cell(95, 10, txt=f"Nose Bleeding: {history.get('NoseBleeding', '')}", ln=0)
    pdf.cell(95, 10, txt=f"When Diagnosed: {history.get('WhenDiagnosed', '')}", ln=1)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Prediction Result", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 12, txt=f"Stage: {history.get('risk_label', '')} ({history.get('confidence', '')}% confidence)", ln=1)
    pdf.ln(5)

    shap_summary = history.get('shap_summary', '')
    if shap_summary:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="AI Explanation (SHAP)", ln=1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, txt=shap_summary)
        pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Health Recommendations", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    for adv in [
        "1. Monitor blood pressure daily and maintain a logbook.",
        "2. Adopt a DASH diet (Dietary Approaches to Stop Hypertension).",
        "3. Exercise at least 30 minutes daily.",
        "4. Reduce sodium intake to less than 2,300 mg per day.",
        "5. Manage stress through meditation, yoga, or deep breathing.",
        "6. Avoid excessive alcohol and smoking.",
        "7. Visit your healthcare provider for regular check-ups."
    ]:
        pdf.multi_cell(0, 8, txt=adv)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_grid_id = fs.put(pdf_bytes, filename=f"report_{pred_id}.pdf", content_type='application/pdf')
    reports_col.insert_one({'prediction_id': pred_id, 'user_id': current_user.id, 'pdf_grid_id': pdf_grid_id, 'generated_at': datetime.utcnow()})

    pdf_path = os.path.join('static', 'reports', f"report_{pred_id}.pdf")
    with open(pdf_path, 'wb') as f:
        f.write(pdf_bytes)

    return send_file(pdf_path, as_attachment=True, download_name=f"Report_{history['date'].strftime('%Y%m%d')}.pdf")


if __name__ == "__main__":
    print("Initializing Healthcare Prediction Service...")
    print("MongoDB: mongodb://localhost:27017/mlde")
    app.run(debug=True, port=5000)
