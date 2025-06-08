# SandClinic 🏥

A Streamlit-based AI tool to predict clinic wait times using real-time inputs and machine learning.

## Features
- Predicts average wait time using Random Forest or Ridge Regression
- Visualizes wait time heatmaps across weekdays and times
- Built with Streamlit, Plotly, scikit-learn

## Files
- `app.py`: Main Streamlit app
- `clinic_data.csv`: Clinic dataset
- `rf_model.joblib`: Random Forest model
- `ridge_model.joblib`: Ridge Regression model
- `notebook.ipynb`: Data exploration and model training
- `requirements.txt`: Required Python packages

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py