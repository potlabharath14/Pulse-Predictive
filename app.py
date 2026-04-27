import os
from flask import Flask
from flask_login import LoginManager

from routes.auth import auth_bp
from routes.prediction import prediction_bp
from models.db import get_user_by_id
from utils.ml import load_ml_model

def create_app():
    app = Flask(__name__)
    # Security: Use environment variable for secret key
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'supersecretkey_change_in_production')
    
    # ─── Flask-Login Setup ──────────────────────────────────────────────────────
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    @login_manager.user_loader
    def load_user(user_id):
        return get_user_by_id(user_id)
        
    # ─── Register Blueprints ───────────────────────────────────────────────────
    app.register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)
    
    # ─── Load ML Model ─────────────────────────────────────────────────────────
    load_ml_model()
    
    return app

app = create_app()

if __name__ == "__main__":
    print("Initializing Healthcare Prediction Service...")
    print("MongoDB: mongodb://localhost:27017/pulse_predictive")
    app.run(debug=True, port=5000)
