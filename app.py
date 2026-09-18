"""
Flask Backend API
E-Commerce Sales, Customer Segmentation & Purchase Prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os
import traceback

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────
# Load models at startup
# ──────────────────────────────────────────────
MODEL_DIR = "models"

def load_models():
    models = {}
    try:
        models["purchase_clf"]   = joblib.load(os.path.join(MODEL_DIR, "purchase_classifier.pkl"))
        models["revenue_reg"]    = joblib.load(os.path.join(MODEL_DIR, "revenue_regressor.pkl"))
        models["kmeans"]         = joblib.load(os.path.join(MODEL_DIR, "customer_segmentation.pkl"))
        models["scaler"]         = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
        models["revenue_scaler"] = joblib.load(os.path.join(MODEL_DIR, "revenue_scaler.pkl"))
        models["rfm_scaler"]     = joblib.load(os.path.join(MODEL_DIR, "rfm_scaler.pkl"))
        models["feature_cols"]   = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
        with open(os.path.join(MODEL_DIR, "cluster_labels.json")) as f:
            models["cluster_labels"] = json.load(f)
        print("[INFO] All models loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Could not load models: {e}")
        print("[INFO] Run the notebook first to generate model files.")
    return models

models = load_models()


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────
FEATURE_COLS = [
    "device_type", "user_type", "marketing_channel", "product_category",
    "unit_price", "quantity", "discount_percent", "discount_amount",
    "pages_viewed", "time_on_site_sec", "added_to_cart", "rating",
    "payment_method", "visit_month", "visit_weekday", "visit_season",
    "session_duration_bucket", "location", "price_per_item",
    "total_potential", "cart_and_viewed", "is_weekend"
]

DURATION_MAP = {"Very Short": 0, "Short": 1, "Medium": 2, "Long": 3, "Very Long": 4}


def build_feature_row(data: dict) -> pd.DataFrame:
    """Build a single-row feature DataFrame from API input."""
    unit_price       = float(data.get("unit_price", 0))
    quantity         = int(data.get("quantity", 1))
    discount_percent = float(data.get("discount_percent", 0))
    pages_viewed     = int(data.get("pages_viewed", 1))
    time_on_site     = float(data.get("time_on_site_sec", 0))
    visit_weekday    = int(data.get("visit_weekday", 0))
    added_to_cart    = int(data.get("added_to_cart", 0))
    dur_raw          = data.get("session_duration_bucket", "Short")
    session_dur      = DURATION_MAP.get(str(dur_raw), int(dur_raw) if str(dur_raw).isdigit() else 1)

    row = {
        "device_type":             int(data.get("device_type", 0)),
        "user_type":               int(data.get("user_type", 0)),
        "marketing_channel":       int(data.get("marketing_channel", 0)),
        "product_category":        int(data.get("product_category", 0)),
        "unit_price":              unit_price,
        "quantity":                quantity,
        "discount_percent":        discount_percent,
        "discount_amount":         float(data.get("discount_amount", 0)),
        "pages_viewed":            pages_viewed,
        "time_on_site_sec":        time_on_site,
        "added_to_cart":           added_to_cart,
        "rating":                  float(data.get("rating", 3)),
        "payment_method":          int(data.get("payment_method", 0)),
        "visit_month":             int(data.get("visit_month", 1)),
        "visit_weekday":           visit_weekday,
        "visit_season":            int(data.get("visit_season", 0)),
        "session_duration_bucket": session_dur,
        "location":                int(data.get("location", 0)),
        "price_per_item":          unit_price * (1 - discount_percent / 100),
        "total_potential":         unit_price * quantity,
        "cart_and_viewed":         added_to_cart * pages_viewed,
        "is_weekend":              1 if visit_weekday >= 5 else 0,
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "project": "E-Commerce ML API",
        "version": "1.0",
        "endpoints": [
            "/api/health",
            "/api/predict_purchase",
            "/api/predict_revenue",
            "/api/segment_customer",
            "/api/dataset_stats",
            "/api/batch_predict"
        ]
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": len(models) > 0,
        "available_models": list(models.keys())
    })


@app.route("/api/predict_purchase", methods=["POST"])
def predict_purchase():
    """Predict whether a customer will make a purchase."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        if "purchase_clf" not in models:
            return jsonify({"error": "Model not loaded. Run the notebook first."}), 503

        X = build_feature_row(data)
        X_scaled = models["scaler"].transform(X)

        prediction = int(models["purchase_clf"].predict(X_scaled)[0])
        probability = float(models["purchase_clf"].predict_proba(X_scaled)[0][1])
        confidence = probability if prediction == 1 else (1 - probability)

        return jsonify({
            "prediction": prediction,
            "will_purchase": bool(prediction),
            "purchase_probability": round(probability, 4),
            "confidence": round(confidence, 4),
            "label": "Will Purchase ✅" if prediction == 1 else "Will NOT Purchase ❌"
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/predict_revenue", methods=["POST"])
def predict_revenue():
    """Predict expected revenue for a session."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        if "revenue_reg" not in models:
            return jsonify({"error": "Model not loaded. Run the notebook first."}), 503

        X = build_feature_row(data)
        X_scaled = models["revenue_scaler"].transform(X)
        predicted_revenue = float(models["revenue_reg"].predict(X_scaled)[0])
        predicted_revenue = max(0.0, predicted_revenue)

        return jsonify({
            "predicted_revenue": round(predicted_revenue, 2),
            "predicted_revenue_formatted": f"₹{predicted_revenue:,.2f}"
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/segment_customer", methods=["POST"])
def segment_customer():
    """Segment a customer based on RFM values."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        if "kmeans" not in models:
            return jsonify({"error": "Model not loaded. Run the notebook first."}), 503

        recency   = float(data.get("recency", 30))
        frequency = float(data.get("frequency", 5))
        monetary  = float(data.get("monetary", 1000))

        rfm_values = np.array([[recency, frequency, monetary]])
        rfm_scaled = models["rfm_scaler"].transform(rfm_values)
        cluster = int(models["kmeans"].predict(rfm_scaled)[0])
        label   = models["cluster_labels"].get(str(cluster), f"Cluster {cluster}")

        segment_insights = {
            "Champions":           "High value, recent, frequent buyers. Reward them!",
            "At-Risk High Value":  "Previously high spenders who haven't purchased recently. Re-engage with offers.",
            "Recent Low Spend":    "Recently active but low spending. Upsell opportunities exist.",
            "Hibernating":         "Low recency, frequency and spend. Win-back campaigns recommended."
        }

        return jsonify({
            "cluster_id":  cluster,
            "segment":     label,
            "rfm_input":   {"recency": recency, "frequency": frequency, "monetary": monetary},
            "insight":     segment_insights.get(label, "Analyze further for targeted marketing.")
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/dataset_stats", methods=["GET"])
def dataset_stats():
    """Return summary statistics from the dataset."""
    try:
        df = pd.read_csv("Ecommerce.csv")
        df["visit_date"] = pd.to_datetime(df["visit_date"], format="%d-%m-%Y", errors="coerce")

        total_sessions   = len(df)
        total_purchases  = int(df["purchased"].sum())
        purchase_rate    = round(df["purchased"].mean() * 100, 2)
        total_revenue    = round(df["revenue"].sum(), 2)
        avg_order_value  = round(df[df["revenue"] > 0]["revenue"].mean(), 2)
        unique_customers = int(df["customer_id"].nunique())
        cart_abandon     = round(df["cart_abandoned"].mean() * 100, 2)

        monthly = df[df["revenue"] > 0].groupby("visit_month")["revenue"].sum()
        monthly_dict = {str(int(k)): round(v, 2) for k, v in monthly.items()}

        category_rev = df.groupby("product_category")["revenue"].sum()
        category_dict = {str(int(k)): round(v, 2) for k, v in category_rev.items()}

        return jsonify({
            "total_sessions":    total_sessions,
            "total_purchases":   total_purchases,
            "purchase_rate_pct": purchase_rate,
            "total_revenue":     total_revenue,
            "avg_order_value":   avg_order_value,
            "unique_customers":  unique_customers,
            "cart_abandon_pct":  cart_abandon,
            "monthly_revenue":   monthly_dict,
            "category_revenue":  category_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    """Batch purchase prediction for multiple sessions."""
    try:
        data = request.get_json(force=True)
        if not data or "sessions" not in data:
            return jsonify({"error": "Provide a JSON body with key 'sessions' as a list"}), 400

        if "purchase_clf" not in models:
            return jsonify({"error": "Model not loaded. Run the notebook first."}), 503

        sessions = data["sessions"]
        results = []
        for i, session in enumerate(sessions):
            X = build_feature_row(session)
            X_scaled = models["scaler"].transform(X)
            pred = int(models["purchase_clf"].predict(X_scaled)[0])
            prob = float(models["purchase_clf"].predict_proba(X_scaled)[0][1])
            results.append({
                "index": i,
                "will_purchase": bool(pred),
                "purchase_probability": round(prob, 4)
            })

        return jsonify({"count": len(results), "predictions": results})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
