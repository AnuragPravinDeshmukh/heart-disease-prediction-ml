"""
Prediction View - Interactive Patient Risk Classification
"""
import streamlit as st
import pandas as pd
from src.prediction import predict_heart_disease

def render_prediction_page():
    st.markdown('<div class="main-title">Interactive Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter patient clinical metrics to calculate model risk classification</div>', unsafe_allow_html=True)
    
    # Presets for easy demonstration
    st.markdown("#### ⚡ Quick Presets (Click to autofill sample profiles)")
    preset_cols = st.columns([1, 1, 3])
    
    if "form_data" not in st.session_state:
        st.session_state["form_data"] = {
            "Age": 45,
            "Sex": "Male",
            "ChestPainType": "ATA",
            "RestingBP": 125,
            "Cholesterol": 210,
            "FastingBS": "No",
            "RestingECG": "Normal",
            "MaxHR": 160,
            "ExerciseAngina": "No",
            "Oldpeak": 0.0,
            "ST_Slope": "Up"
        }
        
    with preset_cols[0]:
        if st.button("🟢 Sample: Low Risk"):
            st.session_state["form_data"] = {
                "Age": 38,
                "Sex": "Female",
                "ChestPainType": "ATA",
                "RestingBP": 115,
                "Cholesterol": 190,
                "FastingBS": "No",
                "RestingECG": "Normal",
                "MaxHR": 175,
                "ExerciseAngina": "No",
                "Oldpeak": 0.0,
                "ST_Slope": "Up"
            }
            st.rerun()
            
    with preset_cols[1]:
        if st.button("🔴 Sample: High Risk"):
            st.session_state["form_data"] = {
                "Age": 62,
                "Sex": "Male",
                "ChestPainType": "ASY",
                "RestingBP": 150,
                "Cholesterol": 275,
                "FastingBS": "Yes",
                "RestingECG": "ST",
                "MaxHR": 115,
                "ExerciseAngina": "Yes",
                "Oldpeak": 2.2,
                "ST_Slope": "Flat"
            }
            st.rerun()

    fd = st.session_state["form_data"]
    
    with st.form("prediction_form"):
        st.markdown("### 1. Patient Demographics & Baseline Vitals")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=int(fd["Age"]), step=1,
                                  help="Patient age in years.")
            sex_choice = st.selectbox("Sex", ["Male", "Female"],
                                      index=0 if fd["Sex"] == "Male" else 1,
                                      help="Biological sex of the patient.")
            
        with col2:
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=70, max_value=230,
                                         value=int(fd["RestingBP"]), step=1,
                                         help="Resting blood pressure measured in mm Hg upon hospital admission.")
            cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=650,
                                          value=int(fd["Cholesterol"]), step=1,
                                          help="Serum cholesterol in mg/dl. Note: Values of 0 represent unrecorded clinical measurements in historical dataset.")
            
        with col3:
            fasting_bs = st.selectbox("Fasting Blood Sugar", ["No (< 120 mg/dl)", "Yes (> 120 mg/dl)"],
                                      index=0 if fd["FastingBS"] == "No" else 1,
                                      help="Fasting blood sugar > 120 mg/dl indicates elevated risk / diabetic threshold.")
            
        st.markdown("---")
        st.markdown("### 2. Cardiovascular Activity & Exercise Response")
        col4, col5, col6 = st.columns(3)
        
        cp_map = {
            "ASY": "ASY (Asymptomatic)",
            "ATA": "ATA (Atypical Angina)",
            "NAP": "NAP (Non-Anginal Pain)",
            "TA": "TA (Typical Angina)"
        }
        cp_list = list(cp_map.values())
        cp_curr = cp_map.get(fd["ChestPainType"], cp_list[0])
        
        with col4:
            chest_pain = st.selectbox("Chest Pain Type", cp_list, index=cp_list.index(cp_curr),
                                      help="ASY: Asymptomatic; ATA: Atypical Angina; NAP: Non-Anginal Pain; TA: Typical Angina.")
            max_hr = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220,
                                     value=int(fd["MaxHR"]), step=1,
                                     help="Maximum heart rate reached during exercise stress testing.")
            
        ecg_map = {
            "Normal": "Normal",
            "ST": "ST (ST-T abnormality)",
            "LVH": "LVH (Left ventricular hypertrophy)"
        }
        ecg_list = list(ecg_map.values())
        ecg_curr = ecg_map.get(fd["RestingECG"], ecg_list[0])
        
        with col5:
            resting_ecg = st.selectbox("Resting Electrocardiogram (ECG)", ecg_list, index=ecg_list.index(ecg_curr),
                                       help="Resting ECG results: Normal, ST-T wave abnormalities, or ventricular hypertrophy.")
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"],
                                           index=0 if fd["ExerciseAngina"] == "No" else 1,
                                           help="Whether physical exertion induces angina symptoms.")
            
        slope_map = {
            "Up": "Up (Upsloping)",
            "Flat": "Flat (Flat)",
            "Down": "Down (Downsloping)"
        }
        slope_list = list(slope_map.values())
        slope_curr = slope_map.get(fd["ST_Slope"], slope_list[0])
        
        with col6:
            oldpeak = st.number_input("Oldpeak (ST Depression in mm)", min_value=-2.0, max_value=6.0,
                                      value=float(fd["Oldpeak"]), step=0.1, format="%.1f",
                                      help="ST depression induced by exercise relative to rest.")
            st_slope = st.selectbox("ST Slope (Peak Exercise)", slope_list, index=slope_list.index(slope_curr),
                                    help="Slope of the peak exercise ST segment: Upsloping, Flat, or Downsloping.")
            
        submit_btn = st.form_submit_button("Predict Heart Disease", use_container_width=True, type="primary")

    if submit_btn:
        parsed_patient = {
            "Age": age,
            "Sex": "M" if sex_choice == "Male" else "F",
            "ChestPainType": chest_pain.split()[0],
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": 1 if "Yes" in fasting_bs else 0,
            "RestingECG": resting_ecg.split()[0],
            "MaxHR": max_hr,
            "ExerciseAngina": "Y" if exercise_angina == "Yes" else "N",
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope.split()[0]
        }
        
        with st.spinner("Executing model inference using pre-trained KNN model..."):
            try:
                res = predict_heart_disease(parsed_patient)
                pred = res["prediction"]
                prob_0 = res["probability_0"]
                prob_1 = res["probability_1"]
                
                st.markdown("### Prediction Results")
                
                if pred == 1:
                    st.error("### Model Prediction: Class 1 - Heart Disease")
                    st.write("The machine-learning model predicted **Class 1 (Heart Disease)** based on the entered clinical parameters.")
                else:
                    st.success("### Model Prediction: Class 0 - No Heart Disease")
                    st.write("The machine-learning model predicted **Class 0 (No Heart Disease)** based on the entered clinical parameters.")
                    
                if prob_0 is not None and prob_1 is not None:
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.metric("Class 0 Probability (No Disease)", f"{prob_0 * 100:.1f}%")
                    with p2:
                        st.metric("Class 1 Probability (Heart Disease)", f"{prob_1 * 100:.1f}%")
                    with p3:
                        st.metric("Risk Assessment Level", res["risk_level"])
                        
                    st.progress(prob_1, text=f"Calculated Cardiac Risk Score: {prob_1 * 100:.1f}%")
                
                with st.expander("View Processed Feature Input"):
                    st.dataframe(res["features"], use_container_width=True)
                    
                st.markdown("""
                <div class="disclaimer-card" style="margin-top:1rem;">
                    <strong>Ethical Healthcare Notice:</strong> This is a statistical model classification generated by a machine-learning algorithm. 
                    It is <strong>not a medical diagnosis</strong>. Clinical evaluations must only be performed by certified healthcare professionals.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during model inference: {str(e)}")
