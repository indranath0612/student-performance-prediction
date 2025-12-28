import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="centered"
)

# Load model
model = joblib.load("student_performance_model.pkl")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50;}
    .stButton button {background-color: #3498db; color: white;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Student Performance Predictor")
st.write("Predict student performance using Machine Learning")

st.divider()

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider("Hours Studied", 0.0, 12.0, 5.0)
    sleep_hours = st.slider("Sleep Hours", 0.0, 10.0, 7.0)
    sample_papers = st.number_input("Sample Papers Practiced", 0, 20, 5)

with col2:
    previous_scores = st.number_input("Previous Scores", 0, 100, 70)
    extra_activity = st.radio("Extracurricular Activities", ["No", "Yes"])

extra_activity = 1 if extra_activity == "Yes" else 0

st.divider()

if st.button("🔍 Predict Performance"):
    input_data = np.array([[hours_studied,
                             previous_scores,
                             sleep_hours,
                             sample_papers,
                             extra_activity]])
    
    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Performance Index: **{prediction:.2f}**")

    if prediction >= 75:
        st.balloons()





