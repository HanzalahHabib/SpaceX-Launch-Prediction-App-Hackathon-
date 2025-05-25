import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import random

# Inject full neon theme, sidebar layout, animated transitions, Orbitron font, and mode toggle
st.set_page_config(page_title="SpaceX Landing Predictor", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    html, body, [class*="css"] {
        background-color: #0d0221;
        color: #e0e0f0;
        font-family: 'Orbitron', sans-serif;
    }

    .stApp {
        background: radial-gradient(ellipse at top left, #16002e, #0d0221 80%);
        color: #e0e0f0;
    }

    h1, h2, h3, h4, .stMarkdown, .stSelectbox label {
        color: #bb00ff;
    }

    .stSidebar {
        background-color: #140029;
        border-right: 2px solid #bb00ff33;
        box-shadow: 5px 0 15px #bb00ff44;
    }

    .stSelectbox div[role="button"] {
        background-color: #24004a;
        border: 1px solid #bb00ff;
        color: #f0f0f0;
        box-shadow: 0 0 8px #bb00ff;
        transition: all 0.2s ease-in-out;
    }

    .stSelectbox div[role="button"]:hover {
        background-color: #32005e;
        border-color: #da70ff;
        transform: scale(1.02);
    }

    .stButton button {
        background: linear-gradient(90deg, #a100ff, #5f00ba);
        border: none;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 12px;
        box-shadow: 0 0 15px #a100ff;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background: linear-gradient(90deg, #5f00ba, #a100ff);
        box-shadow: 0 0 25px #d580ff;
        transform: scale(1.05);
    }

    .stMetric {
        background-color: #1a0030;
        border-left: 5px solid #bb00ff;
        padding: 1em;
        margin-top: 1em;
        border-radius: 10px;
        box-shadow: 0 0 12px #bb00ff66;
        transition: 0.3s ease;
    }

    .stJson {
        background-color: #13002b;
        border-radius: 10px;
        padding: 1em;
        border: 1px solid #bb00ff;
        box-shadow: 0 0 14px #bb00ff88;
        margin-top: 1em;
    }

    .stSubheader {
        color: #d580ff !important;
    }

    .css-1cpxqw2 {
        background-color: #24004a !important;
    }

    .mode-toggle {
        color: #e0ccff;
        font-size: 14px;
        text-align: right;
        margin-bottom: 10px;
    }

    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)



# Load model and encoders
model = pickle.load(open("spacex_landing_model.pkl", "rb"))
label_encoders = joblib.load("spacex_label_encoders.pkl")
feature_columns = pickle.load(open("spacex_features.pkl", "rb"))
full_df = pd.read_csv("spacex_final_dataset.csv")

# Add rocket ID ↔ name mapping
rocket_id_to_name = {
    "5e9d0d95eda69955f709d1eb": "Falcon 1",
    "5e9d0d95eda69973a809d1ec": "Falcon 9",
    "5e9d0d95eda69974db09d1ed": "Falcon Heavy"
}
rocket_name_to_id = {v: k for k, v in rocket_id_to_name.items()}

st.title("🚀 SpaceX Landing Success Prediction")
with st.sidebar:
    st.title("🧠 Control Panel")
    st.markdown("### 🎨 Theme Options")
    dark_mode = st.toggle("🌙 Dark Mode", value=True)

    st.markdown("---")
    st.markdown("🚀 **Developed by:** `Muhammad Hanzlah Habib`")
    st.markdown("👨‍💻 **Hackathon Project**: *AI & Data Science*")
    st.markdown("🔗 [GitHub](https://github.com/) | [LinkedIn](https://linkedin.com/)")

# Helper: decode encoded features
def decode_input(input_df, label_encoders):
    decoded = input_df.copy()
    for col in input_df.columns:
        if col in label_encoders:
            decoded[col] = label_encoders[col].inverse_transform([input_df[col].values[0]])[0]
    # *** FIXED THIS LINE ONLY ***
    if "rocket" in decoded.columns:
        decoded["rocket"] = decoded["rocket"].map(rocket_id_to_name).fillna(decoded["rocket"])
    return decoded

# Helper: get all decoded options for a feature
def get_decoded_options(col):
    if col == "rocket":
        return list(rocket_name_to_id.keys())
    if col in label_encoders:
        return list(label_encoders[col].classes_)
    return sorted(full_df[col].dropna().unique())

# Helper: encode single value
def encode_value(col, val):
    if col == "rocket":
        val = rocket_name_to_id.get(val, val)
    if col in label_encoders:
        return label_encoders[col].transform([val])[0]
    return val

# Features users choose
user_inputs = {}

rocket = st.selectbox("Select Rocket", get_decoded_options("rocket"))
orbit = st.selectbox("Select Orbit", get_decoded_options("orbit"))
launchpad = st.selectbox("Select Launchpad", get_decoded_options("launchpad"))
location = st.selectbox("Select Location", get_decoded_options("location"))
payload_mass_kg = st.selectbox("Select Payload Mass (Kg)", sorted(full_df["payload_mass_kg"].unique()))
booster_version = st.selectbox("Select Booster Version", get_decoded_options("Booster Version"))
payload = st.selectbox("Select Payload", get_decoded_options("Payload"))

# Button to randomize remaining features
if "random_values" not in st.session_state:
    st.session_state.random_values = {}

def randomize_remaining():
    all_cols = set(feature_columns)
    fixed_cols = {"rocket", "orbit", "launchpad", "location", "payload_mass_kg", "Booster Version", "Payload"}
    random_cols = list(all_cols - fixed_cols)

    for col in random_cols:
        st.session_state.random_values[col] = random.choice(get_decoded_options(col))

if st.button("🎲 Randomise"):
    randomize_remaining()

# Build full input row
input_data = {}
input_data["rocket"] = encode_value("rocket", rocket)
input_data["orbit"] = encode_value("orbit", orbit)
input_data["launchpad"] = encode_value("launchpad", launchpad)
input_data["location"] = encode_value("location", location)
input_data["payload_mass_kg"] = float(payload_mass_kg)
input_data["Booster Version"] = encode_value("Booster Version", booster_version)
input_data["Payload"] = encode_value("Payload", payload)

# Add randomized features
for col in feature_columns:
    if col not in input_data:
        rand_val = st.session_state.random_values.get(col, random.choice(get_decoded_options(col)))
        input_data[col] = encode_value(col, rand_val)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
pred_proba = model.predict_proba(input_df)[0][1]  # probability of class 1
prediction = "✅ Successful Landing" if pred_proba > 0.70 else "❌ Landing Failed"

# Decode input for display
decoded_display = decode_input(input_df, label_encoders)

# Results
st.subheader("Prediction Result")
st.metric(label="Landing Prediction", value=prediction)
st.metric(label="Probability of Success", value=f"{pred_proba:.2%}")

st.subheader("🚀 Full Launch Configuration Used")
st.json(decoded_display.to_dict(orient="records")[0])
