import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Student Marks Predictor", layout="centered")
st.title("🎓 Student Marks Prediction (Demo)")

MODEL_PATH = "models/model.pkl"

@st.cache_data
def load_model(path=MODEL_PATH):
    data = joblib.load(path)
    return data['model'], data['features']

try:
    model, features = load_model()
except Exception as e:
    st.error(f"Model not found. Run train_model.py first to create models/model.pkl. Error: {e}")
    st.stop()

mode = st.sidebar.radio("Mode", ["Single prediction", "Batch (CSV upload)"])

if mode == "Single prediction":
    hours = st.number_input("Hours studied per week", min_value=0.0, max_value=100.0, value=8.0, step=0.5)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
    assignments = st.number_input("Assignments submitted (count)", min_value=0, max_value=100, value=6, step=1)
    gpa = st.number_input("Previous GPA", min_value=0.0, max_value=10.0, value=7.0, step=0.01)

    if st.button("Predict"):
        X = pd.DataFrame([[hours, attendance, assignments, gpa]], columns=features)
        pred = model.predict(X)[0]
        st.success(f"Predicted marks: {pred:.2f} / 100")
else:
    st.write("Upload CSV with columns: " + ", ".join(['hours_studied','attendance_pct','assignments_submitted','previous_gpa']))
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in ['hours_studied','attendance_pct','assignments_submitted','previous_gpa'] if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = model.predict(df[['hours_studied','attendance_pct','assignments_submitted','previous_gpa']])
            df_out = df.copy()
            df_out['predicted_marks'] = np.round(preds,2)
            st.dataframe(df_out.head(50))
            csv = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("Download results CSV", data=csv, file_name="predictions.csv", mime="text/csv")
