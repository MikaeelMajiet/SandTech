import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Clinic Wait Time Predictor", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("clinic_data.csv")

@st.cache_resource
def load_models():
    rf_model = joblib.load("rf_model.joblib")
    ridge_pipeline = joblib.load("ridge_model.joblib")
    return rf_model, ridge_pipeline

df = load_data()
rf_model, ridge_model = load_models()

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
time_of_day_options = sorted(df['time_of_day'].dropna().unique())
urban_rural_options = sorted(df['urban_rural'].dropna().unique())
weather_options = sorted(df['weather'].dropna().unique())
clinic_type_options = sorted(df['clinic_type'].dropna().unique())
appointment_type_options = sorted(df['appointment_type'].dropna().unique())
walk_in_policy_options = sorted(df['walk_in_policy'].dropna().unique())

st.sidebar.title("Model & Input Selection")
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Ridge Regression (Scaled)"])

def get_user_inputs():
    doctors = st.sidebar.slider("Doctors", 0, 10, 3)
    nurses = st.sidebar.slider("Nurses", 0, 20, 5)
    staff_count = doctors + nurses

    appointments = st.sidebar.slider("Appointments", 0, 200, 60)
    cancellation_rate = st.sidebar.slider("Cancellation Rate", 0.0, 1.0, 0.1, step=0.01)
    no_show_rate = st.sidebar.slider("No-Show Rate", 0.0, 1.0, 0.1, step=0.01)

    cancellations = int(appointments * cancellation_rate)
    no_shows = int(appointments * no_show_rate)

    walk_ins = st.sidebar.slider("Expected Walk-ins", 0, 50, 10)
    emergencies = st.sidebar.slider("Emergencies", 0, 10, 1)

    patients_served = max(appointments - cancellations - no_shows + walk_ins + emergencies, 1)
    staff_per_patient = round(staff_count / patients_served, 2)
    efficiency_score = round((patients_served / staff_count) - 10 if staff_count else -10, 2)

    return {
        'day_of_week': st.sidebar.selectbox("Day of Week", days_order),
        'time_of_day': st.sidebar.selectbox("Time of Day", time_of_day_options),
        'urban_rural': st.sidebar.selectbox("Urban vs Rural", urban_rural_options),
        'weather': st.sidebar.selectbox("Weather", weather_options),
        'clinic_type': st.sidebar.selectbox("Clinic Type", clinic_type_options),
        'appointment_type': st.sidebar.selectbox("Appointment Type", appointment_type_options),
        'walk_in_policy': st.sidebar.selectbox("Walk-in Policy", walk_in_policy_options),
        'cancellations': cancellations,
        'cancellation_rate': cancellation_rate,
        'no_shows': no_shows,
        'no_show_rate': no_show_rate,
        'staff_count': staff_count,
        'staff_doctors': doctors,
        'staff_nurses': nurses,
        'clinic_open_hours': st.sidebar.slider("Open Hours", 1, 24, 8),
        'appointment_lead_time': st.sidebar.slider("Lead Time (days)", 0, 30, 5),
        'power_outage': st.sidebar.selectbox("Power Outage?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
        'equipment_downtime': st.sidebar.selectbox("Equipment Downtime?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
        'appointments': appointments,
        'walk_ins': walk_ins,
        'emergencies': emergencies,
        'avg_appointment_duration': st.sidebar.slider("Avg Duration (min)", 5, 60, 20),
        'triage_score_avg': st.sidebar.slider("Avg Triage Score", 1.0, 5.0, 2.5, step=0.1),
        'transport_access_score': st.sidebar.slider("Transport Score", 0.0, 1.0, 0.8, step=0.01),
        'patients_served': patients_served,
        'staff_per_patient': staff_per_patient,
        'efficiency_score': efficiency_score
    }

def predict(df, model):
    return model.predict(df)[0]

inputs = get_user_inputs()
input_df = pd.DataFrame([inputs])
model = rf_model if model_choice == "Random Forest" else ridge_model

# Single prediction
predicted_wait = predict(input_df, model)
st.title("📅 Clinic Wait Time Predictor")
st.header(f"🕒 Predicted Avg Wait Time: `{predicted_wait:.2f}` minutes ({model_choice})")

# HEATMAP
st.subheader("📊 Wait Time Heatmap")
heatmap_data = []

for day in days_order:
    for time in time_of_day_options:
        row = inputs.copy()
        row['day_of_week'] = day
        row['time_of_day'] = time
        heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data)
heatmap_df['predicted_wait'] = model.predict(heatmap_df)

heatmap_pivot = heatmap_df.pivot(index='day_of_week', columns='time_of_day', values='predicted_wait')
heatmap_pivot = heatmap_pivot.reindex(index=days_order)

fig = px.imshow(
    heatmap_pivot,
    color_continuous_scale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
    labels=dict(x="Time of Day", y="Day of Week", color="Wait Time (min)"),
    aspect="auto"
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 🚀 Run this app locally:")
st.code("streamlit run app.py", language="bash")