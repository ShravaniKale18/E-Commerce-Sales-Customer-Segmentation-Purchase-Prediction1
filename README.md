# 🛒 E-Commerce Sales, Customer Segmentation & Purchase Prediction

> An end-to-end Machine Learning system built on a 25,000-session e-commerce dataset, featuring a **Flask REST API** backend, a **Streamlit** interactive dashboard, and three ML models: **purchase prediction**, **revenue forecasting**, and **customer segmentation**.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Frontend Screens](#frontend-screens)
- [Results & Metrics](#results--metrics)
- [Technologies Used](#technologies-used)

---

## 📖 Project Overview

This project delivers a complete ML pipeline for an e-commerce platform:

| Task | Model | Output |
|------|-------|--------|
| **Purchase Prediction** | Random Forest Classifier | Binary (Will buy / Won't buy) + probability |
| **Revenue Forecasting** | XGBoost Regressor | Predicted session revenue (₹) |
| **Customer Segmentation** | K-Means Clustering (RFM) | Customer segment label + marketing insight |

---

## 📊 Dataset

**File:** `Ecommerce.csv` — 25,000 rows × 29 columns

| Column | Description |
|--------|-------------|
| `customer_id` | Unique customer identifier |
| `session_id` | Session identifier |
| `visit_date` | Date of visit (DD-MM-YYYY) |
| `device_type` | 0=Desktop, 1=Mobile, 2=Tablet |
| `user_type` | 0=New, 1=Returning |
| `marketing_channel` | 0–5 (Organic/Email/Social/Referral/Direct/Paid) |
| `product_category` | 0–7 product categories |
| `unit_price` | Price of product (₹) |
| `quantity` | Items in session |
| `discount_percent` | Discount applied (%) |
| `revenue` | Actual revenue generated |
| `pages_viewed` | Pages visited in session |
| `time_on_site_sec` | Session duration (seconds) |
| `added_to_cart` | Cart addition flag |
| `purchased` | **Target** — 1=purchased, 0=not purchased |
| `cart_abandoned` | Cart abandonment flag |
| `rating` | Customer rating (1–5) |
| `payment_method` | 0–5 payment types |
| `visit_season` | 0=Spring, 1=Summer, 2=Autumn, 3=Winter |
| `session_duration_bucket` | Very Short / Short / Medium / Long / Very Long |
| `location` | Encoded location code |

---

## 🤖 Machine Learning Models

### 1. Purchase Prediction — Random Forest Classifier
- **Features:** 22 (including 4 engineered features)
- **Train/Test Split:** 80/20, stratified
- **Key hyperparameters:** `n_estimators=200`, `max_depth=12`
- **Metrics:** Accuracy, ROC-AUC, F1-Score, Precision, Recall

### 2. Revenue Forecasting — XGBoost Regressor
- **Target:** `revenue` (only sessions where revenue > 0)
- **Metrics:** RMSE, MAE, R²
- **Key hyperparameters:** `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`

### 3. Customer Segmentation — K-Means (RFM)
- **Input:** Recency, Frequency, Monetary per customer
- **K = 4** clusters (chosen via Elbow Method)
- **Segments:** Champions · At-Risk High Value · Recent Low Spend · Hibernating

### Engineered Features
```python
price_per_item  = unit_price * (1 - discount_percent / 100)
total_potential = unit_price * quantity
cart_and_viewed = added_to_cart * pages_viewed
is_weekend      = 1 if visit_weekday >= 5 else 0
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│               Streamlit Frontend (Port 8501)          │
│  Dashboard | Predict | Revenue | Segment | Analytics  │
└───────────────────────┬──────────────────────────────┘
                        │ HTTP REST (JSON)
┌───────────────────────▼──────────────────────────────┐
│               Flask REST API (Port 5000)              │
│   /api/predict_purchase  /api/predict_revenue         │
│   /api/segment_customer  /api/dataset_stats           │
└───────────────────────┬──────────────────────────────┘
                        │ joblib.load()
┌───────────────────────▼──────────────────────────────┐
│                  models/ Directory                    │
│  purchase_classifier.pkl   revenue_regressor.pkl      │
│  customer_segmentation.pkl feature_scaler.pkl         │
│  revenue_scaler.pkl        rfm_scaler.pkl             │
│  feature_cols.pkl          cluster_labels.json        │
└──────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│            Jupyter Notebook (ecommerce_ml.ipynb)      │
│  EDA → Preprocessing → Training → Evaluation → Save  │
└──────────────────────────────────────────────────────┘
                        │
                 Ecommerce.csv (25K rows)
```

---

## 📁 Project Structure

```
📦 E-Commerce ML Project
├── 📓 ecommerce_ml.ipynb        # Full ML training notebook
├── 🐍 app.py                    # Flask REST API backend
├── 🌐 streamlit_app.py          # Streamlit frontend dashboard
├── 📋 requirements.txt          # Python dependencies
├── 📄 README.md                 # This file
├── 📘 projectfile.docx          # Full project report
├── 📊 Ecommerce.csv             # Dataset (25,000 rows)
└── 📁 models/                   # Saved model artifacts (auto-created)
    ├── purchase_classifier.pkl
    ├── revenue_regressor.pkl
    ├── customer_segmentation.pkl
    ├── feature_scaler.pkl
    ├── revenue_scaler.pkl
    ├── rfm_scaler.pkl
    ├── feature_cols.pkl
    └── cluster_labels.json
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### Step 1 — Clone/open the project folder
```bash
cd "E-Commerce Sales, Customer Segmentation & Purchase Prediction1"
```

### Step 2 — Create virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### Step 1 — Train the ML Models
Open and run all cells in the notebook:
```bash
jupyter notebook ecommerce_ml.ipynb
```
This will create the `models/` directory with all saved artifacts and generate visualisation images.

### Step 2 — Start the Flask Backend
```bash
python app.py
```
Flask API will be available at: `http://127.0.0.1:5000`

### Step 3 — Start the Streamlit Frontend
Open a **new terminal** and run:
```bash
streamlit run streamlit_app.py
```
Dashboard will open at: `http://localhost:8501`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and available routes |
| `GET` | `/api/health` | Health check + model status |
| `POST` | `/api/predict_purchase` | Predict if customer will purchase |
| `POST` | `/api/predict_revenue` | Predict expected revenue |
| `POST` | `/api/segment_customer` | Classify customer into RFM segment |
| `GET` | `/api/dataset_stats` | Aggregate stats from dataset |
| `POST` | `/api/batch_predict` | Batch purchase predictions |

### Example — Purchase Prediction
```bash
curl -X POST http://127.0.0.1:5000/api/predict_purchase \
  -H "Content-Type: application/json" \
  -d '{
    "device_type": 1,
    "user_type": 1,
    "marketing_channel": 2,
    "product_category": 0,
    "unit_price": 1299.0,
    "quantity": 2,
    "discount_percent": 10,
    "pages_viewed": 12,
    "time_on_site_sec": 800,
    "added_to_cart": 1,
    "rating": 4,
    "visit_month": 11,
    "visit_weekday": 5
  }'
```
**Response:**
```json
{
  "prediction": 1,
  "will_purchase": true,
  "purchase_probability": 0.8342,
  "confidence": 0.8342,
  "label": "Will Purchase ✅"
}
```

### Example — Customer Segmentation
```bash
curl -X POST http://127.0.0.1:5000/api/segment_customer \
  -H "Content-Type: application/json" \
  -d '{"recency": 7, "frequency": 15, "monetary": 12000}'
```

---

## 📊 Frontend Screens

| Screen | Description |
|--------|-------------|
| 🏠 **Dashboard** | KPI tiles, monthly revenue trend, category breakdown, gauge charts |
| 🔮 **Purchase Prediction** | Form-based predictor with probability gauge |
| 💰 **Revenue Forecast** | Session revenue estimator with capture rate |
| 👥 **Customer Segmentation** | RFM input form, segment label, radar chart, segment guide |
| 📊 **Data Analytics** | Interactive tabs — sales trends, purchase behaviour, feature explorer |
| ℹ️ **About** | Architecture diagram, model table, usage instructions |

---

## 📈 Results & Metrics

| Model | Metric | Expected Range |
|-------|--------|---------------|
| Random Forest Classifier | Accuracy | ~82–88% |
| Random Forest Classifier | ROC-AUC | ~0.88–0.93 |
| XGBoost Classifier | Accuracy | ~83–89% |
| XGBoost Classifier | ROC-AUC | ~0.89–0.94 |
| XGBoost Regressor | R² | ~0.75–0.85 |
| K-Means Clustering | Silhouette Score | ~0.30–0.50 |

*(Exact metrics are printed when running `ecommerce_ml.ipynb`)*

---

## 🛠️ Technologies Used

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **ML / Data** | scikit-learn, XGBoost, pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn (notebook), Plotly (frontend) |
| **Backend API** | Flask, Flask-CORS |
| **Frontend** | Streamlit |
| **Model Persistence** | joblib |
| **Notebook** | Jupyter |

---

## 👩‍💻 Author
Project built as part of the **E-Commerce Sales, Customer Segmentation & Purchase Prediction** academic/industry project.

---

*Run `ecommerce_ml.ipynb` → `python app.py` → `streamlit run streamlit_app.py` to get started.*
