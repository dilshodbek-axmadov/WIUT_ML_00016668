import streamlit as st 
import joblib
import pandas as pd
from datetime import datetime

# Loading the model
@st.cache_resource
def load_model():
    return joblib.load('traffic_model.pkl')

model_data = load_model()
preprocessor = model_data['preprocessor']
model = model_data['model']

st.title("Metro Interstate Traffic Volume Predictor")
st.write("Enter weather and time details to predict hourly westbound traffic on I-94")

# user inputs (based on the raw data)
col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date", value=datetime.today())
    time = st.time_input("Time", value="12:00", step=60*15)
    temp_c = st.number_input("Temperature (°C)", min_value=-30.0, max_value=37.0, value=20.0)
    rain_1h = st.number_input("Rain in the last hour (mm)", min_value=0.0, max_value=56.0, step=0.1)
    snow_1h = st.number_input("Snow in the last hour (mm)", min_value=0.0, max_value=0.51, value=0.0, step=0.01)

with col2:
    clouds_all = st.slider("Cloud cover (%)", min_value=0, max_value=100, value=50)
    weather_main = st.selectbox("Weather Condition", 
                                ['Clouds', 'Clear', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog',
       'Thunderstorm', 'Snow', 'Squall', 'Smoke'])
    holiday = st.selectbox("Holiday",
                           ['No Holiday', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day',
       'Christmas Day', 'New Years Day', 'Washingtons Birthday',
       'Memorial Day', 'Independence Day', 'State Fair', 'Labor Day',
       'Martin Luther King Jr Day'])
    
# Predict button
if st.button("Predict Traffic Volume"):
    # combine date + time into datetime
    date_time = pd.Timestamp.combine(date, time)

    is_weekend = 1 if date_time.day_of_week>=5 else 0
    # Creating DataFrame matching training features
    input_data = pd.DataFrame({
        'holiday':[holiday],
        'rain_1h':[rain_1h],
        'snow_1h':[snow_1h],
        'clouds_all':[clouds_all/100.0],
        'weather_main':[weather_main],
        'month':[date_time.month],
        'day_of_week':[date_time.day_of_week],
        'is_weekend': [is_weekend],
        'hour':[date_time.hour],
        'temp_c':[temp_c]
    })

    # Applying preprocessing
    X_processed = preprocessor.transform(input_data)

    # Predict the output
    prediction = model.predict(X_processed)[0]

    st.success(f"Predicted Traffic Volume: **{int(prediction):,} vehicles per hour**")

def validate_inputs():
    if rain_1h<0 or snow_1h < 0:
        st.error("Rain and snow cannot be negative.")
        return False
    return True

if not validate_inputs():
    st.stop()