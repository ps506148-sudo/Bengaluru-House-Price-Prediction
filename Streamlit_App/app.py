from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Prediction | Bengaluru", page_icon="🏠", layout="centered")
ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "Model" / "house_price_model_bengaluru.pkl"

@st.cache_resource
def load_artifact():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)

st.title("🏠 House Price Prediction System")
st.write("Machine-learning estimate of historical Bengaluru property listing prices.")
st.caption("Academic demonstration · Not a live 2026 market valuation or verified sale price")
try:
    artifact = load_artifact()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

locations = artifact["locations"]
area_types = artifact["area_types"]
preferred = "whitefield" if "whitefield" in locations else locations[0]
st.header("Enter Property Details")
with st.form("house_form"):
    location = st.selectbox("Bengaluru locality", locations, index=locations.index(preferred), format_func=lambda v: v.title())
    area_type = st.selectbox("Area type", area_types, index=area_types.index("Super built-up  Area") if "Super built-up  Area" in area_types else 0)
    bhk = st.number_input("Bedrooms (BHK)", 1, 12, 2, 1)
    sqft = st.number_input("Area (square feet)", 250.0, 15000.0, 1200.0, 50.0)
    bath = st.number_input("Bathrooms", 0, 20, 2, 1)
    balcony = st.number_input("Balconies", 0, 10, 1, 1)
    submitted = st.form_submit_button("Predict House Price", use_container_width=True)

if submitted:
    X = pd.DataFrame([{"location": location, "area_type": area_type,
                       "bhk": bhk, "sqft": sqft, "bath": bath, "balcony": balcony}])
    try:
        price_lakh = max(0.0, float(artifact["pipeline"].predict(X)[0]))
        price_inr = price_lakh * 100000
        st.success(f"Estimated listing price: ₹{price_lakh:,.2f} lakh")
        st.metric("Equivalent", f"₹{price_inr / 10000000:,.2f} crore" if price_lakh >= 100 else f"₹{price_inr:,.0f}")
        st.caption("Based on historical Bengaluru dataset patterns; actual current prices can vary greatly.")
    except Exception as error:
        st.error(f"Prediction failed: {error}")

with st.expander("About the trained model and evaluation"):
    m = artifact["metrics"]
    st.write(f"Selected algorithm: **{artifact['model_name']}** (compared against regularized Linear Regression and Random Forest)")
    st.write(f"Holdout MAE: **₹{m['MAE_lakh']:.2f} lakh** | RMSE: **₹{m['RMSE_lakh']:.2f} lakh**")
    st.write(f"R²: **{m['R2']:.4f}** | Median absolute percentage error: **{m['Median_APE_percent']:.2f}%**")
    st.info("Metrics refer to a held-out test subset of historical Bengaluru records, not a 2026 market test.")
