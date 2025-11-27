# app.py — CORRECT Gradio 4+ version (works locally + Hugging Face)
import gradio as gr
import joblib
import pandas as pd
from datetime import datetime

# Load model
model_data = joblib.load("traffic_model.pkl")
preprocessor = model_data["preprocessor"]
model = model_data["model"]

def predict_traffic(date_str, time_str, temp_c, rain_1h, snow_1h, clouds_all, weather_main, holiday):
    try:
        # Parse date and time from strings
        date = pd.to_datetime(date_str)
        time = pd.to_datetime(time_str).time()
        date_time = pd.Timestamp.combine(date.date(), time)
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'holiday':       [holiday],
            'rain_1h':       [rain_1h],
            'snow_1h':       [snow_1h],
            'clouds_all':    [clouds_all / 100.0],
            'weather_main':  [weather_main],
            'month':         [date_time.month],
            'day_of_week':   [date_time.weekday()],
            'is_weekend':    [1 if date_time.weekday() >= 5 else 0],
            'hour':          [date_time.hour],
            'temp_c':        [temp_c]
        })
        
        X = preprocessor.transform(input_data)
        prediction = int(model.predict(X)[0])
        return f"**{prediction:,} vehicles per hour** on I-94 westbound"
        
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface — works with Gradio 4+
with gr.Blocks(title="I-94 Traffic Predictor") as demo:
    gr.Markdown("# Metro Interstate Traffic Volume Predictor")
    gr.Markdown("Enter date, time, and weather → get instant traffic prediction")

    with gr.Row():
        with gr.Column():
            date_str = gr.Textbox(label="Date (YYYY-MM-DD)", value=datetime.today().strftime("%Y-%m-%d"))
            time_str = gr.Textbox(label="Time (HH:MM)", value="12:00")
            temp_c = gr.Slider(-30, 37, value=20, step=0.5, label="Temperature (°C)")
            rain_1h = gr.Slider(0, 56, value=0, step=0.1, label="Rain last hour (mm)")
            snow_1h = gr.Slider(0, 0.51, value=0, step=0.01, label="Snow last hour (mm)")

        with gr.Column():
            clouds_all = gr.Slider(0, 100, value=50, step=1, label="Cloud cover (%)")
            weather_main = gr.Dropdown(
                choices=['Clouds', 'Clear', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog',
                         'Thunderstorm', 'Snow', 'Squall', 'Smoke'],
                value='Clouds', label="Weather Condition"
            )
            holiday = gr.Dropdown(
                choices=['No Holiday', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day',
                         'Christmas Day', 'New Years Day', 'Washingtons Birthday',
                         'Memorial Day', 'Independence Day', 'State Fair', 'Labor Day',
                         'Martin Luther King Jr Day'],
                value='No Holiday', label="Holiday"
            )

    btn = gr.Button("Predict Traffic Volume", variant="primary")
    output = gr.Markdown()

    btn.click(
        fn=predict_traffic,
        inputs=[date_str, time_str, temp_c, rain_1h, snow_1h, clouds_all, weather_main, holiday],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()