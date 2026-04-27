from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from models.db import users_col, MongoUser

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user_doc = users_col.find_one({'username': username})
        if user_doc and check_password_hash(user_doc['password'], password):
            login_user(MongoUser(user_doc))
            return redirect(url_for('prediction.home'))
        else:
            flash('Login failed. Check your username and password.')
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Security: Basic sanitization / validation could go here
        if not username or not password:
            flash('Username and password are required.')
            return redirect(url_for('auth.register'))
            
        if users_col.find_one({'username': username}):
            flash('Username already exists.')
            return redirect(url_for('auth.register'))
            
        hashed = generate_password_hash(password)
        result = users_col.insert_one({'username': username, 'password': hashed, 'created_at': datetime.utcnow()})
        user_doc = users_col.find_one({'_id': result.inserted_id})
        login_user(MongoUser(user_doc))
        return redirect(url_for('prediction.home'))
    return render_template('register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('prediction.home'))
