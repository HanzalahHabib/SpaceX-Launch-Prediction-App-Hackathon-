# 🚀 SpaceX Landing Success Prediction

A **Machine Learning-powered web app** that predicts the success of SpaceX rocket landings using mission data. With a **neon-themed modern UI** and real-time interactivity, this app blends aerospace data science with elegant user experience.

![Demo](https://github.com/HanzalahHabib/Spacex-Launch-Prediction-App-Hackathon-/blob/main/demo.gif)

---

## 🎯 Project Overview

This Streamlit-based web application allows users to input key parameters of a SpaceX launch and predicts whether the rocket will **successfully land** or **fail** using a trained ML model. It features dynamic forms, random configuration generation, and a real-time probability meter — all wrapped in a **sleek neon-purple interface**.

---

## 🛠️ Tech Stack

- **Frontend & App**: [Streamlit](https://streamlit.io/)  
- **Machine Learning**: Scikit-learn (Model Training & Prediction)  
- **Data Handling**: Pandas, NumPy  
- **Storage**: Pickle & Joblib (for models and encoders)  
- **Design**: Custom CSS injection for neon aesthetic ✨  

---

## 📊 Features

✅ Predicts landing success with probability  
✅ Beautiful neon purple/pink UI  
✅ Encoded values decoded for display (e.g., Rocket IDs → Rocket Names)  
✅ Random launch config generator 🎲  
✅ Fully interactive and responsive  
✅ Built using real SpaceX launch data  

---

## 🔍 Input Parameters

Users can choose from:

- 🚀 Rocket (Falcon 1, Falcon 9, Falcon Heavy)
- 🛰 Orbit Type
- 📍 Launchpad
- 🌍 Launch Location
- 📦 Payload Mass (kg)
- 💉 Booster Version
- 🔐 Payload Type
- ➕ Randomly assigned features

All values are encoded for ML compatibility and decoded for readability.

---

## 🧠 Machine Learning

- **Model Type**: Classification (e.g., Random Forest / Logistic Regression)
- **Target**: Binary – Successful vs. Failed landing
- **Preprocessing**:
  - Label Encoding for categorical features
  - Feature selection for optimal model performance

---

## 📁 Project Structure

├── app.py # Streamlit app
├── spacex_landing_model.pkl # Trained ML model
├── spacex_label_encoders.pkl # Encoders for categorical variables
├── spacex_features.pkl # Selected model features
├── spacex_final_dataset.csv # Cleaned training dataset
├── README.md # This file


---

## 🧪 How to Run Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/spacex-predictor.git
   cd spacex-predictor

2.Install dependencies

pip install -r requirements.txt

3.Launch the app

streamlit run app.py

5.Open http://localhost:8501 in your browser.

