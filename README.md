# 🫀 Pulse Predictive — Hypertension Stage Prediction System

An end-to-end Machine Learning web application that predicts hypertension stages using patient clinical data. Built with **Flask**, **MongoDB**, **scikit-learn**, and **SHAP explainability**, featuring a premium dark-mode Glassmorphism UI.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-Class Prediction** | Classifies into 4 stages: Normal, Stage-1, Stage-2, Hypertensive Crisis |
| **Explainable AI (SHAP)** | Visual + text explanation of why each prediction was made |
| **User Authentication** | Register/Login system with password hashing |
| **MongoDB Storage** | Users, predictions, reports, and SHAP plots all stored in MongoDB + GridFS |
| **PDF Reports** | Downloadable clinical reports with patient data, AI explanation, and health recommendations |
| **Interactive Dashboard** | Card-based prediction history, Chart.js confidence trend & stage distribution charts |
| **Premium UI** | Dark-mode, glassmorphism, Material Icons, micro-animations, fully responsive |

---

## 📁 Project Structure

```text
Pulse-Predictive/
│
├── static/
│   ├── style.css              # Premium dark-mode Glassmorphism design system
│   ├── shap_plots/            # Auto-generated SHAP explanation images
│   └── reports/               # Auto-generated PDF reports (local cache)
├── templates/
│   ├── index.html             # Prediction form + results page
│   ├── dashboard.html         # History cards, charts, stats
│   ├── login.html             # Authentication - login
│   └── register.html          # Authentication - register
├── eda_plots/                 # Auto-generated EDA visualizations
│   ├── stage_distribution.png
│   ├── gender_vs_stages.png
│   ├── age_vs_stages.png
│   ├── severity_pie.png
│   └── correlation_heatmap.png
├── app.py                     # Flask application (routes, ML, MongoDB, PDF)
├── patient_data.csv           # Real patient dataset (1825 rows, 14 columns)
├── model_training.py          # ML pipeline (cleaning, EDA, training, saving)
├── logreg_model.pkl           # Trained model + encoders + scaler (auto-generated)
├── generate_data.py           # Synthetic data generator (optional/legacy)
└── README.md                  # You are here
```

---

## 🔧 Prerequisites

### 1. Install Python
- Download from [python.org](https://www.python.org/downloads/) or use [Anaconda](https://www.anaconda.com/download/)

### 2. Install MongoDB
- Download from [mongodb.com](https://www.mongodb.com/try/download/community)
- Ensure `mongod` is running on `localhost:27017`

### 3. Install Python Packages

```bash
pip install flask flask-login pymongo gridfs numpy pandas scikit-learn matplotlib seaborn shap fpdf werkzeug
```

---

## 🚀 How to Run

Execute these commands **in order** from the project directory:

### Step 1: Start MongoDB
```bash
mongod
```

### Step 2: Train the ML Model
Cleans the dataset, runs EDA, trains 6 models, selects Logistic Regression, and saves `logreg_model.pkl`.
```bash
python model_training.py
```

### Step 3: Launch the Web App
```bash
python app.py
```

### Step 4: Open in Browser
Navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📊 Dataset: `patient_data.csv`

Real clinical patient data with **1825 records** and **14 columns**:

| Column | Values |
|---|---|
| Gender | Male, Female |
| Age | 18-34, 35-50, 51-64, 65+ |
| History | Yes, No (family history of BP) |
| Patient | Yes, No (currently diagnosed) |
| TakeMedication | Yes, No |
| Severity | Mild, Moderate, Severe |
| BreathShortness | Yes, No |
| VisualChanges | Yes, No |
| NoseBleeding | Yes, No |
| WhenDiagnosed | <1 Year, 1-5 Years, >5 Years |
| Systolic | 100+, 111-120, 121-130, 130+ |
| Diastolic | 70-80, 81-90, 91-100, 100+, 130+ |
| ControlledDiet | Yes, No |
| **Stages (Target)** | **NORMAL, HYPERTENSION (Stage-1), HYPERTENSION (Stage-2), HYPERTENSIVE CRISIS** |

---

## 🤖 ML Pipeline

### Data Cleaning (`model_training.py`)
- Strips whitespace and normalizes inconsistencies (e.g., `Sever` → `Severe`)
- Fixes BP range formats (`121- 130` → `121 - 130`)
- Removes 477 duplicate rows
- LabelEncoder for all categorical features
- MinMaxScaler for feature scaling

### Models Compared
| Model | Accuracy | F1-Score |
|---|---|---|
| Decision Tree | 100.0% | 1.0000 |
| Random Forest | 100.0% | 1.0000 |
| Naive Bayes | 100.0% | 1.0000 |
| **Logistic Regression** ✅ | **94.4%** | **0.9326** |
| SVM | 94.4% | 0.9326 |
| KNN | 93.7% | 0.9263 |

> **Selected: Logistic Regression** — chosen for interpretability, SHAP compatibility, and resistance to overfitting on this dataset. Tree-based models show 100% accuracy likely due to data structure, making LR the safer generalization choice.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask, Flask-Login |
| **Database** | MongoDB (PyMongo + GridFS) |
| **ML** | scikit-learn, SHAP, pandas, numpy |
| **PDF** | FPDF |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JS |
| **Charts** | Chart.js |
| **Icons** | Google Material Icons Round |
| **Fonts** | Outfit (Google Fonts) |

---

## 🧪 Sample Test Cases

### High Risk Input (Hypertensive Crisis)
| Field | Value |
|---|---|
| Gender | Male |
| Age | 65+ |
| Family History | Yes |
| Patient | Yes |
| Medication | Yes |
| Severity | Severe |
| Breath Shortness | Yes |
| Visual Changes | Yes |
| Nose Bleeding | Yes |
| When Diagnosed | >5 Years |
| Systolic | 130+ |
| Diastolic | 100+ |
| Controlled Diet | Yes |
| **Expected** | 🚨 **HYPERTENSIVE CRISIS** |

### Low Risk Input (Stage-1)
| Field | Value |
|---|---|
| Gender | Female |
| Age | 18-34 |
| Family History | Yes |
| Patient | No |
| Medication | No |
| Severity | Mild |
| Breath Shortness | No |
| Visual Changes | No |
| Nose Bleeding | No |
| When Diagnosed | <1 Year |
| Systolic | 111 - 120 |
| Diastolic | 81 - 90 |
| Controlled Diet | No |
| **Expected** | 🟡 **HYPERTENSION (Stage-1)** |

---

## 🗄️ MongoDB Collections

| Collection | Purpose |
|---|---|
| `users` | User accounts (username, hashed password, created_at) |
| `records` | All prediction records with inputs, results, recommendations, SHAP data |
| `reports` | Generated PDF report metadata + GridFS file references |
| `fs.files` / `fs.chunks` | GridFS binary storage for SHAP plots and PDF files |

---

## 📄 License

This project is for educational and research purposes.

---

*Built with ❤️ using Flask, MongoDB, scikit-learn, and SHAP*
