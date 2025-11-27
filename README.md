# I-94 Traffic Volume Predictor

Live app: https://huggingface.co/spaces/dilshod-axmadov/i94-traffic-predictor

A machine learning model that predicts **hourly westbound traffic volume on I-94** (Metro Interstate area) based on weather and time conditions.

### Features used
- Date & Time (hour, day of week, month, weekend or not)  
- Temperature (°C)  
- Rain in last hour (mm)  
- Snow in last hour (mm)  
- Cloud cover (%)  
- Weather condition (Clear, Rain, Snow, Fog, etc.)  
- Holiday (yes/no)

### How to use
1. Choose date and time  
2. Enter current weather conditions  
3. Click **Predict Traffic Volume**  
4. Get instant prediction in vehicles per hour

### Model details
- Trained on the **Metro Interstate Traffic Volume** dataset (UCI)  
- Preprocessing + best-performing model (CatBoost / XGBoost / RandomForest)  
- Features engineered from `date_time` column  
- Model and ColumnTransformer saved with joblib

### Run locally in Jupyter Notebook
If you want to run or explore the model locally, you should install the libraries in the requirements.txt file
But also, I write them in here:
Pandas
Numpy
Maplotlib
Seaborn
Scikit-learn
XGBoost
CatBoost
Gradio