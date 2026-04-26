from app import app
import json

with app.test_client() as client:
    response = client.post('/predict', data={
        'gender': 'Male',
        'age': '45',
        'family_history': 'Yes',
        'medication': 'No',
        'symptoms': 'Headache',
        'lifestyle': 'Active',
        'systolic_bp': '140',
        'diastolic_bp': '90'
    })
    print("STATUS:", response.status_code)
    # just print if there's an error string
    html = response.data.decode('utf-8')
    if "An processing error occurred" in html:
        print("ERROR IN RESPONSE DASHBOARD!")
        for line in html.split('\n'):
            if "An processing error occurred" in line:
                print(line)
    else:
        print("SUCCESS! HTML returned.")
