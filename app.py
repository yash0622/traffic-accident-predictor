import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap

# --- UI SETTINGS ---
st.set_page_config(page_title="Traffic Accident Predictor", page_icon="🚦", layout="wide")

# --- 🎨 ADVANCED CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); color: #ffffff; }
    header {visibility: hidden;} footer {visibility: hidden;}
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #FF416C, #FF4B2B); color: white; border: none; width: 100%; 
        padding: 18px; font-size: 20px; font-weight: 900; border-radius: 50px; transition: all 0.4s ease 0s;
        box-shadow: 0px 5px 15px rgba(255, 75, 43, 0.4); text-transform: uppercase; letter-spacing: 2px;
    }
    div.stButton > button:first-child:hover { transform: translateY(-5px); box-shadow: 0px 10px 25px rgba(255, 75, 43, 0.8); }
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important; backdrop-filter: blur(10px); border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1); color: #00e5ff !important; font-size: 18px !important; font-weight: bold !important;
    }
    div[data-testid="stExpanderDetails"] {
        background: rgba(0, 0, 0, 0.2); border-radius: 0px 0px 12px 12px; border: 1px solid rgba(255, 255, 255, 0.1); border-top: none;
    }
    h1 { font-family: 'Trebuchet MS', sans-serif; text-transform: uppercase; letter-spacing: 2px; color: #00e5ff !important; text-shadow: 0px 0px 10px rgba(0, 229, 255, 0.5); }
    h2, h3 { color: #f8f9fa !important; }
    div[data-testid="stAlert"] { border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.4); border-left: 5px solid; }
    label { color: #a8b2d1 !important; font-weight: 600 !important; }

    /* --- REMOVE TYPING CURSOR FROM DROPDOWNS --- */
    div[data-baseweb="select"] input {
        caret-color: transparent !important; 
        cursor: pointer !important; 
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource 
def load_assets():
    model = joblib.load('outputs/models/best_model.pkl')
    scaler = joblib.load('outputs/models/scaler.pkl')
    label_encoders = joblib.load('outputs/models/label_encoders.pkl')
    feature_columns = joblib.load('outputs/models/feature_columns.pkl')
    return model, scaler, label_encoders, feature_columns

model, scaler, label_encoders, feature_columns = load_assets()

# --- DICTIONARIES ---
india_geography = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Tirupati", "Unknown"],
    "Arunachal Pradesh": ["Itanagar", "Tawang", "Pasighat", "Unknown"],
    "Assam": ["Guwahati", "Silchar", "Dibrugarh", "Jorhat", "Unknown"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Unknown"],
    "Chhattisgarh": ["Raipur", "Bhilai", "Bilaspur", "Korba", "Unknown"],
    "Goa": ["Panaji", "Vasco da Gama", "Margao", "Unknown"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar", "Unknown"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat", "Ambala", "Unknown"],
    "Himachal Pradesh": ["Shimla", "Manali", "Dharamshala", "Solan", "Unknown"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Unknown"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi", "Belagavi", "Unknown"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Unknown"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Unknown"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane", "Nashik", "Aurangabad", "Unknown"],
    "Delhi": ["New Delhi", "Unknown"]
}

time_windows = {
    "12:00 AM to 1:00 AM": 0, "1:00 AM to 2:00 AM": 1, "2:00 AM to 3:00 AM": 2, "3:00 AM to 4:00 AM": 3, 
    "4:00 AM to 5:00 AM": 4, "5:00 AM to 6:00 AM": 5, "6:00 AM to 7:00 AM": 6, "7:00 AM to 8:00 AM": 7, 
    "8:00 AM to 9:00 AM": 8, "9:00 AM to 10:00 AM": 9, "10:00 AM to 11:00 AM": 10, "11:00 AM to 12:00 PM": 11,
    "12:00 PM to 1:00 PM": 12, "1:00 PM to 2:00 PM": 13, "2:00 PM to 3:00 PM": 14, "3:00 PM to 4:00 PM": 15, 
    "4:00 PM to 5:00 PM": 16, "5:00 PM to 6:00 PM": 17, "6:00 PM to 7:00 PM": 18, "7:00 PM to 8:00 PM": 19, 
    "8:00 PM to 9:00 PM": 20, "9:00 PM to 10:00 PM": 21, "10:00 PM to 11:00 PM": 22, "11:00 PM to 12:00 AM": 23
}

# --- HEADER ---
st.title("🚦 Traffic Accident Prediction System")
st.markdown("*Powered by Machine Learning & Historical Road Safety Data*")
st.markdown("---")

# --- SPLIT SCREEN LAYOUT ---
left_panel, space, right_panel = st.columns([1.2, 0.1, 1.5])

with left_panel:
    st.markdown("### ⚙️ ACCIDENT SCENARIO PARAMETERS")

    with st.expander("📍 ACCIDENT LOCATION", expanded=True):
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            state = st.selectbox("STATE", list(india_geography.keys()))
        with col_l2:
            city = st.selectbox("CITY", india_geography[state])

        area = st.selectbox("LOCATION DETAILS", ['Urban Area', 'Rural Area', 'Residential Area', 'Highway', 'Intersection', 'Unknown'])

    with st.expander("🌤️ WEATHER & TIME CONDITIONS", expanded=True):
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            day = st.selectbox("DAY OF WEEK", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            weather = st.selectbox("WEATHER", ['Clear', 'Rainy', 'Foggy', 'Snow', 'Hazy'])
        with col_e2:
            selected_time_window = st.selectbox("TIME OF ACCIDENT", list(time_windows.keys()), index=11)
            hour = time_windows[selected_time_window] 
            light_cond = st.selectbox("LIGHTING", ['Daylight', 'Darkness - lit', 'Darkness - unlit', 'Dusk/Dawn'])

    with st.expander("🛣️ ROAD & VEHICLE DETAILS", expanded=True):
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            road_type = st.selectbox("ROAD CLASSIFICATION", ['National Highway', 'State Highway', 'Expressway', 'City Road', 'Other'])
            road_cond = st.selectbox("ROAD CONDITION", ['Dry', 'Wet', 'Snow/Ice', 'Under Construction'])
            veh_type = st.selectbox("VEHICLE TYPE", ['Car', 'Truck', 'Bus', 'Motorcycle', 'Cycle', 'Pedestrian', 'Auto-Rickshaw'])
        with col_r2:
            num_veh = st.number_input("TOTAL VEHICLES INVOLVED", 1, 10, 2)
            speed_limit = st.number_input("SPEED LIMIT (KM/H)", min_value=10, max_value=500, value=60, step=10)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("PREDICT ACCIDENT SEVERITY 🚀")

with right_panel:
    st.markdown("### 📊 PREDICTION ANALYTICS")

    if predict_btn:
        input_dict = {
            'State Name': state, 'City Name': city,
            'Day of Week': day, 'Hour': hour, 'Weather Conditions': weather, 
            'Speed Limit (km/h)': speed_limit, 'Road Condition': road_cond,
            'Vehicle Type Involved': veh_type, 'Number of Vehicles Involved': num_veh,
            'Accident Location Details': area, 
            'Lighting Conditions': light_cond, 
            'Road Type': road_type,
            'Is_Night': 1 if (hour >= 18 or hour <= 6) else 0,
            'Is_Weekend': 1 if day in ['Saturday', 'Sunday'] else 0
        }

        df_sample = pd.DataFrame([input_dict])
        for col in feature_columns:
            if col not in df_sample.columns: df_sample[col] = 0
        df_sample = df_sample[feature_columns]

        for col, le in label_encoders.items():
            if col in df_sample.columns:
                df_sample[col] = df_sample[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0)

        X_scaled = scaler.transform(df_sample)
        pred = model.predict(X_scaled)[0]

        st.caption(f"📍 **LOCATION:** {city.upper()}, {state.upper()} ({road_type.upper()})")

        if pred == 0:
            st.success("## 🟢 PREDICTED SEVERITY: MINOR / SLIGHT")
            st.write(f"An accident on this type of road under these conditions is most likely to result in minor property damage with no major injuries.")
        elif pred == 1:
            st.warning("## 🟠 PREDICTED SEVERITY: SERIOUS")
            st.write(f"A crash under these conditions is highly likely to be serious, resulting in severe injuries that require hospitalization.")
        else:
            st.error("## 🔴 PREDICTED SEVERITY: FATAL")
            st.write(f"CRITICAL WARNING: The model predicts a FATAL severity for this scenario. High speeds and these conditions strongly correlate with loss of life.")

        st.markdown("---")

        st.markdown("#### 🧠 ACCIDENT RISK FACTORS")

        with st.spinner("Analyzing AI Prediction Data..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)

            if isinstance(shap_values, list): pred_shap_values = shap_values[pred][0]
            elif len(np.array(shap_values).shape) == 3: pred_shap_values = shap_values[0, :, pred]
            else: pred_shap_values = shap_values[0]

            pred_shap_values = np.array(pred_shap_values).flatten()

            impacts = list(zip(feature_columns, pred_shap_values))
            impacts.sort(key=lambda x: x[1], reverse=True)

            def format_feature(feature_name):
                val = input_dict.get(feature_name, "Unknown")
                if feature_name == "Hour": val = selected_time_window
                elif feature_name == "Is_Night": val = "Yes" if val == 1 else "No"
                elif feature_name == "Is_Weekend": val = "Yes" if val == 1 else "No"
                return f"**{feature_name}** ({val})"

            risk_factors = [format_feature(f) for f, v in impacts if v > 0.02][:3]
            safety_factors = [format_feature(f) for f, v in impacts if v < -0.02][-3:]
            safety_factors.reverse() 

            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.error("🚨 **SEVERITY INCREASING FACTORS:**")
                if risk_factors:
                    for f in risk_factors: st.write(f"🔺 {f}")
                else:
                    st.write("None Detected")

            with col_res2:
                st.success("🛡️ **SEVERITY DECREASING FACTORS:**")
                if safety_factors:
                    for f in safety_factors: st.write(f"🔽 {f}")
                else:
                    st.write("None Detected")
    else:
        st.info("👈 System ready. Configure the accident parameters and click Predict to analyze severity.")
