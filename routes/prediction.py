import traceback
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, send_file
from flask_login import login_required, current_user
from bson.objectid import ObjectId

from models.db import records_col
from utils.ml import predict_patient
from utils.pdf import generate_pdf_report
from utils.recommendations import get_recommendations

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/')
def home():
    return render_template('index.html')

@prediction_bp.route('/dashboard')
@login_required
def dashboard():
    history = list(records_col.find({'user_id': current_user.id}).sort('date', -1))
    return render_template('dashboard.html', history=history)

@prediction_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Capture form inputs and sanitize
            form_data = {
                'Gender': str(request.form['gender']).strip(),
                'Age': str(request.form['age']).strip(),
                'History': str(request.form['history']).strip(),
                'Patient': str(request.form['patient']).strip(),
                'TakeMedication': str(request.form['take_medication']).strip(),
                'Severity': str(request.form['severity']).strip(),
                'BreathShortness': str(request.form['breath_shortness']).strip(),
                'VisualChanges': str(request.form['visual_changes']).strip(),
                'NoseBleeding': str(request.form['nose_bleeding']).strip(),
                'WhenDiagnosed': str(request.form['when_diagnosed']).strip(),
                'Systolic': str(request.form['systolic']).strip(),
                'Diastolic': str(request.form['diastolic']).strip(),
                'ControlledDiet': str(request.form['controlled_diet']).strip(),
            }

            # Predict
            result = predict_patient(form_data)
            
            # Combine form_data and result to generate recommendations
            prediction_data = {**form_data, 'risk_class': result['stage_info']['class']}
            recommendations = get_recommendations(prediction_data)

            # Save to MongoDB
            history_record_id = None
            if current_user.is_authenticated:
                doc = {
                    'user_id': current_user.id,
                    'date': datetime.utcnow(),
                    **form_data,
                    'predicted_stage': result['predicted_stage'],
                    'confidence': result['confidence'],
                    'risk_class': result['stage_info']['class'],
                    'risk_label': result['stage_info']['label'],
                    'shap_plot_path': result['shap_plot_path'],
                    'shap_summary': result['shap_summary'],
                    'shap_grid_id': result['shap_grid_id'],
                    'recommendations': recommendations
                }
                ins = records_col.insert_one(doc)
                history_record_id = str(ins.inserted_id)

            return render_template('index.html',
                                   predicted_stage=result['predicted_stage'],
                                   stage_info=result['stage_info'],
                                   confidence=result['confidence'],
                                   prob_breakdown=result['prob_breakdown'],
                                   shap_plot=result['shap_plot_path'],
                                   shap_summary=result['shap_summary'],
                                   recommendations=recommendations,
                                   history_id=history_record_id)

        except Exception as e:
            traceback.print_exc()
            return render_template('index.html', error=f"A processing error occurred: {str(e)}")

    return render_template('index.html')


@prediction_bp.route('/delete_prediction/<pred_id>')
@login_required
def delete_prediction(pred_id):
    try:
        records_col.delete_one({'_id': ObjectId(pred_id), 'user_id': current_user.id})
    except Exception:
        pass
    return redirect(url_for('prediction.dashboard'))


@prediction_bp.route('/api/chart_data')
@login_required
def chart_data():
    history = list(records_col.find({'user_id': current_user.id}).sort('date', 1))
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


@prediction_bp.route('/download_pdf/<pred_id>')
@login_required
def download_pdf(pred_id):
    pdf_path, history = generate_pdf_report(pred_id, current_user.id, current_user.username)
    if not pdf_path:
        return "Not found", 404
        
    return send_file(pdf_path, as_attachment=True, download_name=f"Report_{history['date'].strftime('%Y%m%d')}.pdf")
