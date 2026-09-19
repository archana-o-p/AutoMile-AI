import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==============================================================================
# 1. PAGE SETUP & THEME-COMPATIBLE STYLING
# ==============================================================================
st.set_page_config(
    page_title="AutoMile AI | Fuel Efficiency Platform",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling designed to look clean in both Light and Dark mode
st.markdown("""
<style>
    /* Header Card */
    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .hero-header {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0;
    }
    .hero-desc {
        color: #94a3b8 !important;
        font-size: 0.95rem;
        margin-top: 6px;
    }

    /* Telemetry Table/Row Styling with high contrast */
    .telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        margin-bottom: 8px;
        background-color: #f1f5f9;
        border-radius: 8px;
        border-left: 4px solid #0284c7;
    }
    .telemetry-label {
        font-weight: 600;
        color: #475569 !important;
        font-size: 0.92rem;
    }
    .telemetry-value {
        font-weight: 700;
        color: #0f172a !important;
        font-size: 0.95rem;
    }

    /* Result Card */
    .result-container {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        border-radius: 14px;
        padding: 28px 20px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 8px 20px rgba(2, 132, 199, 0.25);
    }
    .result-label {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #e0f2fe;
    }
    .result-number {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 6px 0;
    }
    .result-sub {
        color: #bae6fd;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CACHED ASSETS (MODEL & DATASET)
# ==============================================================================
@st.cache_resource
def load_production_model():
    if not os.path.exists("model.pkl"):
        st.error("❌ `model.pkl` not found in current directory.")
        st.stop()
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_source_dataset():
    if not os.path.exists("cardekho_dataset.csv"):
        st.error("❌ `cardekho_dataset.csv` not found in current directory.")
        st.stop()
    return pd.read_csv("cardekho_dataset.csv")

model = load_production_model()
df_ref = load_source_dataset()

# ==============================================================================
# 3. SIDEBAR: 7 VEHICLE ATTRIBUTES
# ==============================================================================
st.sidebar.markdown("### ⚙️ Vehicle Specifications")
st.sidebar.caption("Configure vehicle parameters below:")

# Brand
brand_options = sorted(df_ref["brand"].dropna().unique().tolist())
default_brand_index = brand_options.index("Maruti") if "Maruti" in brand_options else 0
brand = st.sidebar.selectbox("Brand Name", brand_options, index=default_brand_index)

# Model
model_options = sorted(df_ref[df_ref["brand"] == brand]["model"].dropna().unique().tolist())
car_model = st.sidebar.selectbox("Model Variant", model_options)

# Engine
matched_car_records = df_ref[(df_ref["brand"] == brand) & (df_ref["model"] == car_model)]
engine_default = int(matched_car_records["engine"].median()) if not matched_car_records.empty else 1197

engine = st.sidebar.number_input(
    "Engine Capacity (CC)",
    min_value=600,
    max_value=7000,
    value=engine_default,
    step=50
)

# Fuel & Transmission
fuel_type = st.sidebar.selectbox("Fuel Category", ["Petrol", "Diesel", "CNG", "LPG", "Electric"], index=0)
transmission_type = st.sidebar.selectbox("Transmission System", ["Manual", "Automatic"], index=0)

# Age & KM Driven
vehicle_age = st.sidebar.slider("Vehicle Age (Years)", min_value=0, max_value=30, value=5, step=1)
km_driven = st.sidebar.number_input(
    "Total Distance Run (KM)",
    min_value=100,
    max_value=1500000,
    value=40000,
    step=2500
)

# ==============================================================================
# 4. BACKGROUND FEATURE ALIGNMENT ENGINE (163 VECTORS)
# ==============================================================================
def resolve_supporting_features(df, b, m, eng, f):
    match = df[(df["brand"] == b) & (df["model"] == m) & (df["engine"] == eng) & (df["fuel_type"] == f)]
    if match.empty:
        match = df[(df["brand"] == b) & (df["model"] == m) & (df["engine"] == eng)]
    if match.empty:
        match = df[(df["brand"] == b) & (df["model"] == m)]
    if match.empty:
        match = df[df["brand"] == b]
    if match.empty:
        match = df

    pwr = float(match["max_power"].mode()[0]) if not match["max_power"].empty else float(match["max_power"].median())
    sts = int(match["seats"].mode()[0]) if not match["seats"].empty else 5
    prc = float(match["selling_price"].median()) if not match["selling_price"].empty else 400000.0
    stype = str(match["seller_type"].mode()[0]) if not match["seller_type"].empty else "Individual"

    return pwr, sts, prc, stype

derived_power, derived_seats, derived_price, derived_seller = resolve_supporting_features(
    df_ref, brand, car_model, engine, fuel_type
)

def construct_inference_matrix():
    feature_row = {feature_name: 0.0 for feature_name in model.feature_names_in_}
    feature_row["vehicle_age"] = float(vehicle_age)
    feature_row["km_driven"] = float(km_driven)
    feature_row["engine"] = float(engine)
    feature_row["max_power"] = float(derived_power)
    feature_row["seats"] = float(derived_seats)
    feature_row["selling_price"] = float(derived_price)

    for prefix, value in [
        ("brand", brand),
        ("model", car_model),
        ("fuel_type", fuel_type),
        ("transmission_type", transmission_type),
        ("seller_type", derived_seller)
    ]:
        col_key = f"{prefix}_{value}"
        if col_key in feature_row:
            feature_row[col_key] = 1.0

    return pd.DataFrame([feature_row], columns=model.feature_names_in_)

# ==============================================================================
# 5. MAIN PRESENTATION CONSOLE
# ==============================================================================
st.markdown("""
<div class="hero-card">
    <h1 class="hero-header">🚘 Automotive Mileage Predictor</h1>
    <div class="hero-desc">
        Real-time vehicle fuel efficiency estimation platform powered by Machine Learning.
    </div>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.subheader("Vehicle Telemetry Summary")
    st.write("Verify the provided telemetry parameters prior to running inference:")

    # Clean, high-contrast rows that display clearly in light or dark theme
    st.markdown(f"""
    <div class="telemetry-row">
        <span class="telemetry-label">Manufacturer & Model</span>
        <span class="telemetry-value">{brand} {car_model}</span>
    </div>
    <div class="telemetry-row">
        <span class="telemetry-label">Engine Displacement</span>
        <span class="telemetry-value">{engine:,} CC</span>
    </div>
    <div class="telemetry-row">
        <span class="telemetry-label">Fuel Category</span>
        <span class="telemetry-value">{fuel_type}</span>
    </div>
    <div class="telemetry-row">
        <span class="telemetry-label">Transmission System</span>
        <span class="telemetry-value">{transmission_type}</span>
    </div>
    <div class="telemetry-row">
        <span class="telemetry-label">Vehicle Age</span>
        <span class="telemetry-value">{vehicle_age} Year(s)</span>
    </div>
    <div class="telemetry-row">
        <span class="telemetry-label">Odometer Reading</span>
        <span class="telemetry-value">{km_driven:,} km</span>
    </div>
    <div class="telemetry-row">
        <span class="telemetry-label">Calibrated Engine Power</span>
        <span class="telemetry-value">{derived_power:.1f} bhp</span>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    predict_action = st.button("🚀 Calculate Estimated Mileage", type="primary", use_container_width=True)

with right_col:
    st.subheader("Inference Result")

    if predict_action:
        # Run model inference
        matrix_row = construct_inference_matrix()
        predicted_mileage = float(model.predict(matrix_row)[0])
        unit_label = "km/kg" if fuel_type in ["CNG", "LPG"] else "km/l"

        # Main Professional Result Card
        st.markdown(
            f"""
        <div class="result-container">
            <div class="result-label">Predicted Mileage</div>
            <div class="result-number">{predicted_mileage:.2f} <span style="font-size:1.5rem;">{unit_label}</span></div>
            <div class="result-sub">Computed for <b>{brand} {car_model}</b> ({engine:,} CC {fuel_type})</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.write("")

        # City vs Highway mileage estimates (Real-world value add)
        city_mileage = predicted_mileage * 0.85
        highway_mileage = predicted_mileage * 1.10

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label="Estimated City Driving",
                value=f"{city_mileage:.2f} {unit_label}",
                help="Accounts for typical stop-and-go urban traffic conditions (~15% lower).",
            )
        with col_m2:
            st.metric(
                label="Estimated Highway Cruising",
                value=f"{highway_mileage:.2f} {unit_label}",
                help="Accounts for steady highway speed and high-gear efficiency (~10% higher).",
            )

    else:
        st.info(
            " Verify vehicle telemetry on the left and click **Calculate"
            " Estimated Mileage**."
        )