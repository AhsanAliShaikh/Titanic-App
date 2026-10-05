import streamlit as st
import joblib
import numpy as np

# Page settings
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Model load
model = joblib.load("NB_titanic_model.joblib")

# Title
st.title("🚢 Titanic Survival Predictor")
st.markdown("### Will this passenger survive?")
st.write("Passenger ki details enter karein aur prediction dekhein.")

st.divider()

# Inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=10, value=0)

with col2:
    parch = st.number_input("Parents / Children", min_value=0, max_value=10, value=0)
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox("Port of Embarkation", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
    alone = st.selectbox("Traveling Alone?", ["Yes", "No"])

# Encoding (same as notebook)
sex_encoded = 1 if sex == "Male" else 0
embarked_encoded = {
    "Cherbourg (C)": 0,
    "Queenstown (Q)": 1,
    "Southampton (S)": 2
}[embarked]
alone_encoded = 1 if alone == "Yes" else 0

# Predict button
if st.button("Predict Survival", type="primary", use_container_width=True):

    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded, alone_encoded]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()

    if prediction == 1:
        st.success("✅ **Survived!**")
        st.metric("Survival Chance", f"{probability[1]*100:.1f}%")
        st.balloons()
    else:
        st.error("❌ **Did Not Survive**")
        st.metric("Survival Chance", f"{probability[1]*100:.1f}%")

    with st.expander("Input Details"):
        st.write({
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked,
            "Alone": alone
        })

st.divider()
st.caption("Model: Gaussian Naive Bayes | Accuracy ≈ 77.5%")