def get_recommendations(prediction_data):
    """
    Generate personalized health recommendations based on patient data and prediction.
    """
    recs = []
    
    # Extract data safely
    risk_class = prediction_data.get('risk_class', 'success') # success, warning, danger, crisis
    systolic_str = str(prediction_data.get('Systolic', ''))
    diastolic_str = str(prediction_data.get('Diastolic', ''))
    age_group = str(prediction_data.get('Age', ''))
    breath_shortness = str(prediction_data.get('BreathShortness', 'No'))
    visual_changes = str(prediction_data.get('VisualChanges', 'No'))
    nose_bleeding = str(prediction_data.get('NoseBleeding', 'No'))
    controlled_diet = str(prediction_data.get('ControlledDiet', 'No'))
    
    # Base recommendations by risk level
    if risk_class == 'success':
        recs.append("Maintain your current healthy lifestyle and continue routine checkups.")
        recs.append("Aim for at least 150 minutes of moderate aerobic exercise per week.")
        if controlled_diet == 'No':
            recs.append("Consider adopting a balanced, heart-healthy diet to proactively maintain normal blood pressure.")
    
    elif risk_class == 'warning':
        recs.append("Monitor your blood pressure daily at home and maintain a logbook.")
        recs.append("Adopt a DASH diet (Dietary Approaches to Stop Hypertension) focusing on fruits, vegetables, and low-fat dairy.")
        recs.append("Limit sodium intake to less than 2,300 mg (about 1 teaspoon of salt) per day.")
        recs.append("Engage in regular physical activity, at least 30 minutes on most days.")
        
    elif risk_class == 'danger':
        recs.append("Schedule an appointment with your healthcare provider promptly to discuss blood pressure management.")
        recs.append("Strictly reduce sodium intake to ideally 1,500 mg per day.")
        recs.append("Take prescribed medications exactly as directed by your doctor. Do not skip doses.")
        recs.append("Implement stress management techniques such as meditation, deep breathing, or yoga.")
        
    elif risk_class == 'crisis':
        recs.append("URGENT: Seek immediate emergency medical assistance. Your blood pressure is critically high.")
        recs.append("Do not wait to see if your pressure comes down on its own.")
        recs.append("Rest quietly and try to remain calm while waiting for medical help.")
    
    # Conditional recommendations based on inputs
    
    # BP specific
    if '140' in systolic_str or '160' in systolic_str or '100' in diastolic_str:
        if risk_class not in ['danger', 'crisis']:
            recs.append("Your blood pressure numbers indicate a significant elevation. Consult a doctor for a proper evaluation.")
            
    # Age specific
    if age_group in ['50-60', '60-70', '70+']:
        recs.append("Given your age group, it is highly recommended to have comprehensive cardiovascular screenings annually.")
        
    # Symptom specific
    symptoms = []
    if breath_shortness.lower() == 'yes': symptoms.append("shortness of breath")
    if visual_changes.lower() == 'yes': symptoms.append("visual changes")
    if nose_bleeding.lower() == 'yes': symptoms.append("nosebleeds")
    
    if symptoms and risk_class != 'crisis':
        symptom_str = ", ".join(symptoms)
        recs.append(f"You reported {symptom_str}. These can be signs of hypertension complications. Please mention them to your doctor.")
        
    # Formatting
    formatted_recs = [f"{i+1}. {rec}" for i, rec in enumerate(recs)]
    return formatted_recs
