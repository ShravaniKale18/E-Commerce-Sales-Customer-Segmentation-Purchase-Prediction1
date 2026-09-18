"""
Streamlit Frontend
E-Commerce Sales, Customer Segmentation & Purchase Prediction
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce ML Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

FLASK_URL = "http://127.0.0.1:5000"

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .prediction-success {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #155724;
    }
    .prediction-fail {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #721c24;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #343a40;
        border-left: 4px solid #667eea;
        padding-left: 0.7rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def api_call(endpoint: str, method: str = "GET", payload: dict = None):
    try:
        url = f"{FLASK_URL}/{endpoint.lstrip('/')}"
        if method == "GET":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"API Error {resp.status_code}: {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to Flask backend. Please start app.py first."
    except Exception as e:
        return None, str(e)


def gauge_chart(value: float, title: str, color: str = "#667eea") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 40], "color": "#f8d7da"},
                {"range": [40, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#d4edda"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75, "value": value * 100
            }
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=80)
    st.markdown("## 🛒 E-Commerce ML")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "🔮 Purchase Prediction", "💰 Revenue Forecast",
         "👥 Customer Segmentation", "📊 Data Analytics", "ℹ️ About"],
        index=0
    )
    st.markdown("---")
    st.markdown("**Backend Status**")
    health, err = api_call("api/health")
    if health:
        st.success("🟢 Flask API Online")
        models_loaded = health.get("models_loaded", False)
        if models_loaded:
            st.info("✅ ML Models Loaded")
        else:
            st.warning("⚠️ Models not loaded\nRun notebook first")
    else:
        st.error("🔴 Flask API Offline")
    st.markdown("---")
    st.caption("E-Commerce ML v1.0 | Built with Streamlit + Flask")


# ══════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">🛒 E-Commerce Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sales · Customer Segmentation · Purchase Prediction</div>', unsafe_allow_html=True)

    stats, err = api_call("api/dataset_stats")

    if err:
        st.error(err)
    elif stats:
        # KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🛍️ Total Sessions", f"{stats['total_sessions']:,}")
        with col2:
            st.metric("✅ Total Purchases", f"{stats['total_purchases']:,}")
        with col3:
            st.metric("📈 Purchase Rate", f"{stats['purchase_rate_pct']}%")
        with col4:
            st.metric("💰 Total Revenue", f"₹{stats['total_revenue']:,.0f}")
        with col5:
            st.metric("🎯 Avg Order Value", f"₹{stats['avg_order_value']:,.2f}")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-title">📅 Monthly Revenue Trend</div>', unsafe_allow_html=True)
            month_names = {
                "1":"Jan","2":"Feb","3":"Mar","4":"Apr","5":"May","6":"Jun",
                "7":"Jul","8":"Aug","9":"Sep","10":"Oct","11":"Nov","12":"Dec"
            }
            monthly = stats.get("monthly_revenue", {})
            months = [month_names.get(k, k) for k in sorted(monthly.keys(), key=int)]
            revenues = [monthly[k] for k in sorted(monthly.keys(), key=int)]

            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Scatter(
                x=months, y=revenues, mode="lines+markers+text",
                fill="tozeroy", line=dict(color="#667eea", width=3),
                marker=dict(size=8, color="#764ba2"),
                text=[f"₹{v/1000:.0f}K" for v in revenues],
                textposition="top center"
            ))
            fig_monthly.update_layout(
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="Revenue (₹)", xaxis_title="Month",
                plot_bgcolor="#fafafa", paper_bgcolor="white"
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">🏷️ Revenue by Product Category</div>', unsafe_allow_html=True)
            cat_map = {
                "0":"Electronics","1":"Clothing","2":"Home & Garden",
                "3":"Sports","4":"Books","5":"Toys","6":"Beauty","7":"Other"
            }
            cat_rev = stats.get("category_revenue", {})
            cat_labels = [cat_map.get(str(k), f"Cat {k}") for k in sorted(cat_rev.keys(), key=lambda x: int(x))]
            cat_values = [cat_rev[k] for k in sorted(cat_rev.keys(), key=lambda x: int(x))]

            fig_cat = px.bar(
                x=cat_labels, y=cat_values,
                color=cat_values, color_continuous_scale="Viridis",
                labels={"x": "Category", "y": "Revenue (₹)"}
            )
            fig_cat.update_layout(
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False, coloraxis_showscale=False,
                plot_bgcolor="#fafafa", paper_bgcolor="white"
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        # Additional KPI row
        col_c, col_d, col_e = st.columns(3)
        with col_c:
            st.metric("👤 Unique Customers", f"{stats['unique_customers']:,}")
        with col_d:
            st.metric("🛒 Cart Abandon Rate", f"{stats['cart_abandon_pct']}%")
        with col_e:
            purchase_pct = stats['purchase_rate_pct']
            fig_g = gauge_chart(purchase_pct / 100, "Purchase Conversion Rate", "#667eea")
            st.plotly_chart(fig_g, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2: PURCHASE PREDICTION
# ══════════════════════════════════════════════
elif page == "🔮 Purchase Prediction":
    st.markdown('<div class="main-header">🔮 Purchase Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict whether a customer will make a purchase in this session</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('<div class="section-title">🛍️ Session Details</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            device_type = st.selectbox("Device Type", [0, 1, 2],
                format_func=lambda x: {0: "Desktop", 1: "Mobile", 2: "Tablet"}[x])
            user_type = st.selectbox("User Type", [0, 1],
                format_func=lambda x: {0: "New Visitor", 1: "Returning Visitor"}[x])
            marketing_channel = st.selectbox("Marketing Channel", list(range(6)),
                format_func=lambda x: {0:"Organic",1:"Email",2:"Social",3:"Referral",4:"Direct",5:"Paid"}[x])
            payment_method = st.selectbox("Payment Method", list(range(6)),
                format_func=lambda x: {0:"Credit Card",1:"Debit Card",2:"UPI",3:"Wallet",4:"Net Banking",5:"COD"}[x])

        with col2:
            product_category = st.selectbox("Product Category", list(range(8)),
                format_func=lambda x: {0:"Electronics",1:"Clothing",2:"Home",3:"Sports",4:"Books",5:"Toys",6:"Beauty",7:"Other"}[x])
            unit_price = st.number_input("Unit Price (₹)", min_value=1.0, max_value=50000.0, value=999.0, step=50.0)
            quantity = st.number_input("Quantity", min_value=1, max_value=20, value=2)
            discount_percent = st.slider("Discount %", 0, 50, 10)

        with col3:
            pages_viewed = st.number_input("Pages Viewed", min_value=1, max_value=50, value=8)
            time_on_site = st.number_input("Time on Site (sec)", min_value=10, max_value=3600, value=600)
            added_to_cart = st.selectbox("Added to Cart?", [0, 1],
                format_func=lambda x: {0: "No", 1: "Yes"}[x])
            rating = st.slider("Rating", 1, 5, 4)

        st.markdown('<div class="section-title">📅 Visit Details</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)

        with col4:
            visit_month = st.selectbox("Visit Month", list(range(1, 13)),
                format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
            visit_weekday = st.selectbox("Visit Weekday", list(range(7)),
                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

        with col5:
            visit_season = st.selectbox("Season", [0, 1, 2, 3],
                format_func=lambda x: {0:"Spring",1:"Summer",2:"Autumn",3:"Winter"}[x])
            session_duration = st.selectbox("Session Duration",
                ["Very Short", "Short", "Medium", "Long", "Very Long"], index=2)

        with col6:
            location = st.number_input("Location Code", min_value=0, max_value=300, value=50)
            discount_amount = unit_price * quantity * discount_percent / 100

            st.markdown(f"**Calculated Discount Amount:** ₹{discount_amount:.2f}")

        submitted = st.form_submit_button("🔮 Predict Purchase", use_container_width=True)

    if submitted:
        payload = {
            "device_type": device_type, "user_type": user_type,
            "marketing_channel": marketing_channel, "product_category": product_category,
            "unit_price": unit_price, "quantity": quantity,
            "discount_percent": discount_percent, "discount_amount": discount_amount,
            "pages_viewed": pages_viewed, "time_on_site_sec": time_on_site,
            "added_to_cart": added_to_cart, "rating": rating,
            "payment_method": payment_method, "visit_month": visit_month,
            "visit_weekday": visit_weekday, "visit_season": visit_season,
            "session_duration_bucket": session_duration, "location": location
        }

        with st.spinner("Predicting..."):
            result, err = api_call("api/predict_purchase", "POST", payload)

        if err:
            st.error(err)
        elif result:
            st.markdown("---")
            col_res1, col_res2 = st.columns([2, 1])

            with col_res1:
                if result["will_purchase"]:
                    st.markdown(f'<div class="prediction-success">{result["label"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-fail">{result["label"]}</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <br>
                <table style="width:100%;border-collapse:collapse;font-size:1rem;">
                <tr style="background:#f8f9fa;"><td style="padding:8px;font-weight:600;">Purchase Probability</td>
                    <td style="padding:8px;">{result['purchase_probability']*100:.1f}%</td></tr>
                <tr><td style="padding:8px;font-weight:600;">Confidence</td>
                    <td style="padding:8px;">{result['confidence']*100:.1f}%</td></tr>
                <tr style="background:#f8f9fa;"><td style="padding:8px;font-weight:600;">Prediction</td>
                    <td style="padding:8px;">{'Purchase' if result['will_purchase'] else 'No Purchase'}</td></tr>
                </table>
                """, unsafe_allow_html=True)

            with col_res2:
                fig_gauge = gauge_chart(
                    result["purchase_probability"], "Purchase Probability",
                    "#28a745" if result["will_purchase"] else "#dc3545"
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Also show revenue forecast
            st.markdown('<div class="section-title">💰 Expected Revenue Forecast</div>', unsafe_allow_html=True)
            rev_result, rev_err = api_call("api/predict_revenue", "POST", payload)
            if rev_result:
                st.success(f"**Predicted Revenue:** {rev_result['predicted_revenue_formatted']}")


# ══════════════════════════════════════════════
# PAGE 3: REVENUE FORECAST
# ══════════════════════════════════════════════
elif page == "💰 Revenue Forecast":
    st.markdown('<div class="main-header">💰 Revenue Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict expected revenue for a given session using XGBoost Regressor</div>', unsafe_allow_html=True)

    with st.form("revenue_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            r_unit_price = st.number_input("Unit Price (₹)", min_value=1.0, max_value=50000.0, value=1500.0)
            r_quantity   = st.number_input("Quantity", min_value=1, max_value=20, value=3)
            r_discount   = st.slider("Discount %", 0, 50, 15)
            r_product_cat = st.selectbox("Product Category", list(range(8)),
                format_func=lambda x: {0:"Electronics",1:"Clothing",2:"Home",3:"Sports",4:"Books",5:"Toys",6:"Beauty",7:"Other"}[x])

        with col2:
            r_device = st.selectbox("Device", [0,1,2], format_func=lambda x:{0:"Desktop",1:"Mobile",2:"Tablet"}[x])
            r_user_type = st.selectbox("User Type", [0,1], format_func=lambda x:{0:"New",1:"Returning"}[x])
            r_pages = st.number_input("Pages Viewed", min_value=1, max_value=50, value=12)
            r_time  = st.number_input("Time on Site (sec)", min_value=10, max_value=3600, value=900)

        with col3:
            r_cart   = st.selectbox("Added to Cart?", [0,1], format_func=lambda x:{0:"No",1:"Yes"}[x], index=1)
            r_month  = st.selectbox("Month", list(range(1,13)), format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
            r_channel = st.selectbox("Marketing Channel", list(range(6)),
                format_func=lambda x:{0:"Organic",1:"Email",2:"Social",3:"Referral",4:"Direct",5:"Paid"}[x])
            r_payment = st.selectbox("Payment Method", list(range(6)),
                format_func=lambda x:{0:"Credit",1:"Debit",2:"UPI",3:"Wallet",4:"Net Banking",5:"COD"}[x])

        r_submit = st.form_submit_button("💰 Forecast Revenue", use_container_width=True)

    if r_submit:
        payload = {
            "device_type": r_device, "user_type": r_user_type,
            "marketing_channel": r_channel, "product_category": r_product_cat,
            "unit_price": r_unit_price, "quantity": r_quantity,
            "discount_percent": r_discount,
            "discount_amount": r_unit_price * r_quantity * r_discount / 100,
            "pages_viewed": r_pages, "time_on_site_sec": r_time,
            "added_to_cart": r_cart, "rating": 4,
            "payment_method": r_payment, "visit_month": r_month,
            "visit_weekday": 2, "visit_season": 0,
            "session_duration_bucket": "Long", "location": 50
        }
        with st.spinner("Forecasting revenue..."):
            result, err = api_call("api/predict_revenue", "POST", payload)

        if err:
            st.error(err)
        elif result:
            st.markdown("---")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("💰 Predicted Revenue", result["predicted_revenue_formatted"])
            with col_r2:
                potential = r_unit_price * r_quantity
                st.metric("📊 Full Potential Revenue", f"₹{potential:,.2f}")
            with col_r3:
                if potential > 0:
                    capture = result["predicted_revenue"] / potential * 100
                    st.metric("📈 Revenue Capture Rate", f"{min(capture, 100):.1f}%")


# ══════════════════════════════════════════════
# PAGE 4: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.markdown('<div class="main-header">👥 Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">RFM-based K-Means clustering — identify customer segments for targeted marketing</div>', unsafe_allow_html=True)

    # RFM explanation
    with st.expander("📖 What is RFM Analysis?", expanded=False):
        st.markdown("""
        **RFM Analysis** is a customer segmentation technique based on three factors:
        - **R - Recency**: How recently did the customer make a purchase? (Lower = better)
        - **F - Frequency**: How often do they purchase? (Higher = better)
        - **M - Monetary**: How much do they spend? (Higher = better)
        
        The K-Means algorithm groups customers into 4 segments based on these scores.
        """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">📥 Enter Customer RFM Data</div>', unsafe_allow_html=True)
        with st.form("rfm_form"):
            recency   = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
            frequency = st.number_input("Frequency (total sessions)", min_value=1, max_value=100, value=5)
            monetary  = st.number_input("Monetary (total spend ₹)", min_value=0.0, max_value=100000.0, value=2500.0)
            rfm_submit = st.form_submit_button("👥 Identify Segment", use_container_width=True)

    if rfm_submit:
        with st.spinner("Segmenting customer..."):
            result, err = api_call("api/segment_customer", "POST",
                                   {"recency": recency, "frequency": frequency, "monetary": monetary})
        if err:
            st.error(err)
        elif result:
            with col2:
                st.markdown('<div class="section-title">🎯 Segment Result</div>', unsafe_allow_html=True)
                segment_colors = {
                    "Champions": "#28a745",
                    "At-Risk High Value": "#fd7e14",
                    "Recent Low Spend": "#17a2b8",
                    "Hibernating": "#6c757d"
                }
                color = segment_colors.get(result["segment"], "#667eea")
                st.markdown(f"""
                <div style="background:{color}22;border:2px solid {color};border-radius:12px;
                            padding:1.5rem;text-align:center;">
                    <h2 style="color:{color};margin:0;">🏷️ {result['segment']}</h2>
                    <p style="color:#333;margin-top:0.5rem;">Cluster ID: {result['cluster_id']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:#f8f9fa;border-radius:8px;padding:1rem;margin-top:1rem;">
                    <strong>💡 Insight:</strong> {result['insight']}
                </div>
                """, unsafe_allow_html=True)

                # RFM radar chart
                fig_radar = go.Figure(go.Scatterpolar(
                    r=[recency, frequency, monetary / 1000],
                    theta=["Recency", "Frequency", "Monetary (K)"],
                    fill="toself", line_color=color, name="Customer"
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    height=300, margin=dict(l=20, r=20, t=40, b=20),
                    title="RFM Profile"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

    # Segment reference table
    st.markdown('<div class="section-title">📋 Segment Reference Guide</div>', unsafe_allow_html=True)
    seg_df = pd.DataFrame({
        "Segment":       ["Champions", "At-Risk High Value", "Recent Low Spend", "Hibernating"],
        "Recency":       ["Very Recent", "Long Ago", "Recent", "Long Ago"],
        "Frequency":     ["High", "Medium–High", "Low", "Low"],
        "Monetary":      ["High", "High", "Low", "Low"],
        "Strategy":      [
            "Reward & retain — loyalty programs",
            "Re-engage with win-back campaigns",
            "Upsell & cross-sell opportunities",
            "Aggressive discount / re-activation"
        ]
    })
    st.dataframe(seg_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# PAGE 5: DATA ANALYTICS
# ══════════════════════════════════════════════
elif page == "📊 Data Analytics":
    st.markdown('<div class="main-header">📊 Data Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore the e-commerce dataset with interactive visualizations</div>', unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        df = pd.read_csv("Ecommerce.csv")
        df["visit_date"] = pd.to_datetime(df["visit_date"], format="%d-%m-%Y", errors="coerce")
        duration_map = {"Very Short": 0, "Short": 1, "Medium": 2, "Long": 3, "Very Long": 4}
        df["session_duration_num"] = df["session_duration_bucket"].map(duration_map)
        return df

    df = load_data()

    tab1, tab2, tab3 = st.tabs(["📈 Sales Analysis", "🛒 Purchase Behaviour", "🔍 Feature Explorer"])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Revenue by Season")
            season_map = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}
            season_rev = df.groupby("visit_season")["revenue"].sum().reset_index()
            season_rev["Season"] = season_rev["visit_season"].map(season_map)
            fig = px.pie(season_rev, values="revenue", names="Season",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Sales by Day of Week")
            day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
            day_rev = df.groupby("visit_weekday")["revenue"].sum().reset_index()
            day_rev["Day"] = day_rev["visit_weekday"].map(day_map)
            fig = px.bar(day_rev, x="Day", y="revenue",
                         color="revenue", color_continuous_scale="Blues")
            fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Revenue Heatmap: Month × Weekday")
        pivot = df[df["revenue"] > 0].pivot_table(
            values="revenue", index="visit_month", columns="visit_weekday",
            aggfunc="sum", fill_value=0
        )
        pivot.columns = [["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][c] for c in pivot.columns]
        fig_heat = px.imshow(pivot, color_continuous_scale="YlOrRd",
                             labels=dict(x="Day of Week", y="Month", color="Revenue"))
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab2:
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Purchase Rate by Device Type")
            dev_map = {0:"Desktop",1:"Mobile",2:"Tablet"}
            dev_purchase = df.groupby("device_type")["purchased"].mean().reset_index()
            dev_purchase["Device"] = dev_purchase["device_type"].map(dev_map)
            fig = px.bar(dev_purchase, x="Device", y="purchased",
                         color="purchased", color_continuous_scale="Greens",
                         labels={"purchased":"Purchase Rate"})
            fig.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            st.subheader("Cart Abandonment by Marketing Channel")
            ch_map = {0:"Organic",1:"Email",2:"Social",3:"Referral",4:"Direct",5:"Paid"}
            ch_cart = df.groupby("marketing_channel")["cart_abandoned"].mean().reset_index()
            ch_cart["Channel"] = ch_cart["marketing_channel"].map(ch_map)
            fig = px.bar(ch_cart, x="Channel", y="cart_abandoned",
                         color="cart_abandoned", color_continuous_scale="Reds",
                         labels={"cart_abandoned":"Abandon Rate"})
            fig.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Time on Site vs Revenue (Scatter)")
        sample = df[df["revenue"] > 0].sample(min(1000, len(df[df["revenue"] > 0])), random_state=42)
        fig_scatter = px.scatter(
            sample, x="time_on_site_sec", y="revenue",
            color="product_category", size="quantity",
            opacity=0.6, labels={"time_on_site_sec":"Time on Site (s)", "revenue":"Revenue"},
            color_continuous_scale="Viridis"
        )
        fig_scatter.update_layout(height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.subheader("Custom Feature Distribution")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Select Feature", numeric_cols)
        split_by = st.selectbox("Colour by", ["None", "purchased", "device_type", "user_type"])

        if split_by == "None":
            fig = px.histogram(df, x=selected_col, nbins=50, color_discrete_sequence=["#667eea"])
        else:
            fig = px.histogram(df, x=selected_col, color=split_by.replace("_", " "), nbins=50,
                               barmode="overlay", opacity=0.7)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 6: ABOUT
# ══════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        This project builds an end-to-end Machine Learning solution for an E-Commerce dataset with **25,000 sessions** across multiple dimensions.

        ### 🤖 ML Models
        | Model | Task | Algorithm |
        |-------|------|-----------|
        | Purchase Predictor | Binary Classification | Random Forest |
        | Revenue Forecaster | Regression | XGBoost |
        | Customer Segmenter | Clustering | K-Means (RFM) |

        ### 🔬 Dataset Features
        - 29 raw features per session
        - Engineered features: `price_per_item`, `total_potential`, `cart_and_viewed`, `is_weekend`
        - Target: `purchased` (binary), `revenue` (continuous)
        """)

    with col2:
        st.markdown("""
        ### 🏗️ Architecture
        ```
        ┌─────────────────┐
        │  Streamlit UI   │  ← You are here
        └────────┬────────┘
                 │ HTTP REST
        ┌────────▼────────┐
        │   Flask API     │  ← app.py (port 5000)
        └────────┬────────┘
                 │ joblib
        ┌────────▼────────┐
        │   ML Models     │  ← models/ folder
        │  (Random Forest │
        │   XGBoost, KMeans│
        └─────────────────┘
        ```

        ### 📁 Project Files
        | File | Description |
        |------|-------------|
        | `ecommerce_ml.ipynb` | Model training notebook |
        | `app.py` | Flask REST API backend |
        | `streamlit_app.py` | This Streamlit frontend |
        | `models/` | Saved model artifacts |
        | `Ecommerce.csv` | Dataset (25K rows) |

        ### ▶️ Running the Project
        ```bash
        # 1. Train models
        jupyter notebook ecommerce_ml.ipynb

        # 2. Start Flask API
        python app.py

        # 3. Start Streamlit
        streamlit run streamlit_app.py
        ```
        """)

    st.markdown("---")
    st.success("✅ Built with Python · Streamlit · Flask · Scikit-learn · XGBoost · Plotly")
