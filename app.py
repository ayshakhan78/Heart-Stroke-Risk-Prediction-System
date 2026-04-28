import streamlit as st
import pandas as pd
import joblib 

st.set_page_config(page_title="Heart Risk Predictor", page_icon="❤️", layout="centered")

models=joblib.load("LogisticsRgression_Heart.pkl")
scaler=joblib.load("scaler.pkl")
expexcted_columns=joblib.load("columns.pkl")

st.title("🧠❤️ Heart Stroke Risk Prediction System")
st.info("Model: Logistic Regression trained on heart dataset")

st.markdown("---")
st.markdown("### Fill patient details below 👇")

col1, col2 = st.columns(2)

with col1:
    age= st.slider("Age",18,100,40)
    sex=st.selectbox("Sex",["M","F"])
    chest_pain=st.selectbox("Chest Pain Type ",["ATA","NAP","TA","ASY"])
    resting_bp =st.number_input("Restin Bloos Pressure (mm Hg)",80,200,120)
    cholesterol = st.number_input("Cholesterol (mg/dL)",100,600,200)
    fasting_bs=st.selectbox("Fasting Blood Sugar",[0,1])

with col2:
    resting_ecg=st.selectbox("Restign ECG",["Normal","ST","LVH"])
    max_hr=st.slider("Max Heart Rate",60,220,150)
    exercise_angina=st.selectbox("Exercise-Induced Angina",["Y","N"])
    oldpeak=st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0)
    st_slope=st.selectbox("ST Slope",["Up","Flat","Down"])

st.markdown("---")

if st.button("🔍 Predict Risk"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    input_df=pd.DataFrame([raw_input])

    for col in expexcted_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df=input_df[expexcted_columns]

    scaled_input=scaler.transform(input_df)

    prediction=models.predict(scaled_input)[0]



    prob = models.predict_proba(scaled_input)[0][1]

    st.subheader("📊 Risk Analysis")
    st.metric("Risk Score", f"{prob*100:.2f}%")

    
    st.progress(int(prob * 100))

    st.markdown("## 🩺 Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

st.markdown("---")
st.markdown("Made with ❤️ by Ayesha")