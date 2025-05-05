import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import uuid
from datetime import datetime, timedelta

app = Flask(__name__)

# Initialize models and vectorizers
models = {
    'severity': None,
    'affected_devices': None,
    'exploitability': None
}
vectorizers = {
    'vulnerability_type': None,
    'impact': None
}
label_encoders = {}

# Store active scans
active_scans = {}

def load_models():
    """Load pre-trained models and vectorizers"""
    try:
        models['severity'] = joblib.load('models/severity_model.pkl')
        models['affected_devices'] = joblib.load('models/devices_model.pkl')
        models['exploitability'] = joblib.load('models/exploitability_model.pkl')
        vectorizers['vulnerability_type'] = joblib.load('models/vulnerability_vectorizer.pkl')
        vectorizers['impact'] = joblib.load('models/impact_vectorizer.pkl')
        label_encoders['severity'] = joblib.load('models/severity_encoder.pkl')
        label_encoders['devices'] = joblib.load('models/devices_encoder.pkl')
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

def preprocess_input(vulnerability_type, impact, year=2024, count=1):
    """Preprocess input data for prediction"""
    # Transform text features
    vuln_type_tfidf = vectorizers['vulnerability_type'].transform([vulnerability_type])
    impact_tfidf = vectorizers['impact'].transform([impact])
    
    # Combine features
    X_numeric = np.array([[year, count]])
    X_processed = np.hstack([vuln_type_tfidf.toarray(), impact_tfidf.toarray(), X_numeric])
    
    return X_processed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/database')
def database():
    return render_template('database.html')

@app.route('/scanner')
def scanner():
    return render_template('scanner.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['vulnerability_type', 'impact']
        if not all(field in data for field in required_fields):
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
        
        # Preprocess input
        X = preprocess_input(
            data['vulnerability_type'],
            data['impact'],
            data.get('year', 2024),
            data.get('count', 1)
        )
        
        # Make predictions
        severity_pred = models['severity'].predict(X)
        devices_pred = models['affected_devices'].predict(X)
        exploitability_pred = models['exploitability'].predict(X)
        
        # Decode predictions
        severity = label_encoders['severity'].inverse_transform(severity_pred)[0]
        devices = label_encoders['devices'].inverse_transform(devices_pred)[0]
        
        return jsonify({
            'status': 'success',
            'predictions': {
                'severity': severity,
                'affected_devices': devices,
                'exploitability_score': float(exploitability_pred[0])
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/vulnerabilities', methods=['GET'])
def get_vulnerabilities():
    try:
        df = pd.read_csv('vulnerabilities.csv')
        vulnerabilities = df.to_dict('records')
        return jsonify(vulnerabilities)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/vulnerabilities/<int:vulnerability_id>', methods=['GET'])
def get_vulnerability(vulnerability_id):
    try:
        df = pd.read_csv('vulnerabilities.csv')
        vulnerability = df.iloc[vulnerability_id].to_dict()
        return jsonify(vulnerability)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/scan', methods=['POST'])
def start_scan():
    try:
        data = request.get_json()
        scan_id = str(uuid.uuid4())
        
        # Initialize scan
        active_scans[scan_id] = {
            'status': 'running',
            'progress': 0,
            'start_time': datetime.now(),
            'target': data['targetUrl'],
            'type': data['targetType'],
            'scan_type': data['scanType'],
            'checks': data.get('checks', [])
        }
        
        return jsonify({'scan_id': scan_id})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/scan/<scan_id>', methods=['GET'])
def get_scan_status(scan_id):
    try:
        if scan_id not in active_scans:
            return jsonify({'status': 'error', 'message': 'Scan not found'}), 404
            
        scan = active_scans[scan_id]
        
        # Simulate scan progress
        if scan['status'] == 'running':
            scan['progress'] = min(100, scan['progress'] + 10)
            
            if scan['progress'] == 100:
                scan['status'] = 'completed'
                scan['results'] = simulate_scan_results(scan)
        
        return jsonify(scan)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def simulate_scan_results(scan):
    """Simulate scan results for demonstration purposes"""
    results = []
    
    # Common vulnerabilities based on scan type
    if scan['scan_type'] == 'quick':
        vulnerabilities = [
            {
                'title': 'Missing Security Headers',
                'severity': 'Medium',
                'location': scan['target'],
                'description': 'Security headers are missing or misconfigured',
                'recommendation': 'Implement proper security headers'
            }
        ]
    elif scan['scan_type'] == 'full':
        vulnerabilities = [
            {
                'title': 'SQL Injection Vulnerability',
                'severity': 'High',
                'location': f"{scan['target']}/api/users",
                'description': 'SQL injection vulnerability detected in user API endpoint',
                'recommendation': 'Implement parameterized queries'
            },
            {
                'title': 'Cross-Site Scripting (XSS)',
                'severity': 'Medium',
                'location': f"{scan['target']}/search",
                'description': 'Reflected XSS vulnerability in search functionality',
                'recommendation': 'Implement proper input sanitization'
            }
        ]
    else:  # custom scan
        vulnerabilities = []
        if 'sql' in scan['checks']:
            vulnerabilities.append({
                'title': 'SQL Injection Vulnerability',
                'severity': 'High',
                'location': f"{scan['target']}/api/users",
                'description': 'SQL injection vulnerability detected',
                'recommendation': 'Implement parameterized queries'
            })
        if 'xss' in scan['checks']:
            vulnerabilities.append({
                'title': 'Cross-Site Scripting (XSS)',
                'severity': 'Medium',
                'location': f"{scan['target']}/search",
                'description': 'Reflected XSS vulnerability',
                'recommendation': 'Implement proper input sanitization'
            })
        if 'csrf' in scan['checks']:
            vulnerabilities.append({
                'title': 'CSRF Protection Missing',
                'severity': 'Medium',
                'location': scan['target'],
                'description': 'CSRF protection is not implemented',
                'recommendation': 'Implement CSRF tokens'
            })
        if 'auth' in scan['checks']:
            vulnerabilities.append({
                'title': 'Weak Authentication',
                'severity': 'High',
                'location': f"{scan['target']}/login",
                'description': 'Weak password policy and session management',
                'recommendation': 'Implement strong authentication measures'
            })
    
    return vulnerabilities

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    try:
        df = pd.read_csv('vulnerabilities.csv')
        
        # Calculate statistics
        total_vulnerabilities = len(df)
        critical_vulnerabilities = len(df[df['Severity'] == 'Critical'])
        high_vulnerabilities = len(df[df['Severity'] == 'High'])
        avg_exploitability = df['Exploitability Score'].mean()
        
        # Severity distribution
        severity_distribution = {
            'labels': df['Severity'].value_counts().index.tolist(),
            'values': df['Severity'].value_counts().values.tolist()
        }
        
        # Devices distribution
        devices_distribution = {
            'labels': df['Affected Devices'].value_counts().index.tolist(),
            'values': df['Affected Devices'].value_counts().values.tolist()
        }
        
        # Trends over time
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')
        trends = []
        for severity in ['Critical', 'High', 'Medium']:
            severity_data = df[df['Severity'] == severity]
            trends.append({
                'name': severity,
                'dates': severity_data['Year'].dt.strftime('%Y').tolist(),
                'counts': severity_data.groupby('Year').size().tolist()
            })
        
        return jsonify({
            'total_vulnerabilities': total_vulnerabilities,
            'critical_vulnerabilities': critical_vulnerabilities,
            'high_vulnerabilities': high_vulnerabilities,
            'avg_exploitability': avg_exploitability,
            'severity_distribution': severity_distribution,
            'devices_distribution': devices_distribution,
            'trends': trends
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load models
    if not load_models():
        print("Models not found. Please train the models first.")
    
    app.run(debug=True)