import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Domestic Tourism Expenditure Predictor",
    page_icon="🧳",
    layout="wide",
)

# Simple styling for a cleaner, consistent interface
st.markdown(
    """
    <style>
        .block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem;}
        .hero {padding: 1.4rem 1.5rem; border-radius: 16px;
               background: linear-gradient(120deg, #eef5ff, #f5f8fc);
               border: 1px solid #dce7f5; margin-bottom: 1.2rem;}
        .hero h1 {margin: 0 0 .45rem 0; color: #17365d; font-size: 2.15rem;}
        .hero p {margin: 0; color: #43536a; font-size: 1.02rem;}
        .section-card {padding: 1.2rem; border: 1px solid #e2e8f0;
                       border-radius: 14px; background: #ffffff;}
        div.stButton > button {border-radius: 9px; min-height: 2.8rem;
                              font-weight: 600; padding: 0.45rem 1.2rem;}
        [data-testid="stMetric"] {padding: 1rem; border: 1px solid #dce7f5;
                                 border-radius: 12px; background: #f7faff;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Load model --------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "expenditure_model.pkl",
    BASE_DIR / "project_files" / "expenditure_model.pkl",
]

MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.is_file()), None)

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

# -------------------- Header --------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧳 Domestic Tourism Expenditure Predictor</h1>
        <p>Estimate domestic tourism expenditure using survey data and a machine learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)



if MODEL_PATH is None:
    st.error(
        "The trained model file was not found. Place expenditure_model.pkl "
        "in the same folder as app.py, or inside the project_files folder."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "The model could not be loaded. Check that the Python and scikit-learn "
        "versions match the environment used to train/save the model."
    )
    st.caption(f"Technical detail: {exc}")
    st.stop()

st.divider()

# -------------------- Prediction form --------------------
st.subheader("Enter survey details")
st.write("Enter the sector, state code, and district code for the area you want to estimate.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        sector_label = st.selectbox("Sector", ["Rural", "Urban"])

    with col2:
        state_code = st.text_input(
            "State code",
            placeholder="e.g. 09",
            help="Enter the survey state code. Keep leading zeros if present.",
        ).strip()


    with col3:
        district_code = st.text_input(
            "District code",
            placeholder="e.g. 07",
            help="Enter the survey district code. Keep leading zeros if present.",
        ).strip()

    # -------- Additional dummy travel questions --------
    st.divider()
    st.subheader("Travel Preferences")

    travel_col1, travel_col2 = st.columns(2)

    with travel_col1:
        travel_days = st.slider(
            "Number of days travelling",
            min_value=1,
            max_value=30,
            value=3,
        )

        transport_mode = st.selectbox(
            "Mode of transport",
            ["Bus", "Train", "Car", "Flight"],
        )

        transport_hours = st.slider(
            "Transport time (one way, in hours)",
            min_value=0,
            max_value=24,
            value=4,
        )

        num_travellers = st.number_input(
            "Number of travellers",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
        )

    with travel_col2:
        hotel_stars = st.selectbox(
            "Hotel category",
            ["Budget / No-star", "3-star", "4-star", "5-star"],
        )

        food_preference = st.selectbox(
            "Food preference",
            ["Budget meals", "Standard meals", "Premium dining"],
        )

        travel_purpose = st.selectbox(
            "Purpose of travel",
            ["Tourism", "Business", "Education", "Personal visit"],
        )

        activities = st.multiselect(
            "Activities planned",
            ["Sightseeing", "Shopping", "Adventure", "Pilgrimage", "Museums"],
        )

    submitted = st.form_submit_button(
        "Predict expenditure",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not state_code or not district_code:
        st.warning("Please enter both the state code and district code.")
    elif not state_code.isdigit() or not district_code.isdigit():
        st.warning("State and district codes should contain digits only.")
    else:
        # Keep values as strings so codes such as 09 and 07 retain leading zeros.
        input_data = pd.DataFrame(
            {
                "sector": ["1" if sector_label == "Rural" else "2"],
                "state": [state_code],
                "district": [district_code],
            }
        )

        try:
            prediction = float(model.predict(input_data)[0])
            prediction = max(0.0, prediction)

            st.success("Prediction completed")
            st.metric("Estimated total expenditure", f"₹{prediction:,.2f}")
            st.caption(
                "This estimate is generated by the trained model and may be inaccurate, "
                "particularly for combinations that were uncommon or absent in the training data."
            )
        except Exception as exc:
            st.error("Prediction failed for these inputs. Check the codes and try again.")
            st.caption(f"Technical detail: {exc}")

st.divider()

# -------------------- About --------------------
st.subheader("About this project")
st.write(
    "This project uses a machine learning regression pipeline to estimate total "
    "domestic tourism expenditure from sector, state, and district codes in survey data."
)
st.write(
    "Model evaluation showed limited predictive performance. The project is a learning "
    "prototype; further feature engineering and validation are needed before its estimates "
    "can be relied upon for real-world decisions."
)

with st.expander("How to use this tool"):
    st.markdown(
        """
        1. Choose **Rural** or **Urban**.
        2. Enter the state and district codes used in the survey dataset.
        3. Select **Predict expenditure** to view the model estimate.

        The codes are survey identifiers, not free-text state or district names.
        """
    )

