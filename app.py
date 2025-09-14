# app.py
import os
import sys
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)
app = application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                wind_speed=float(request.form.get('wind_speed')),
                wind_direction=float(request.form.get('wind_direction')),
                theoretical_power=float(request.form.get('theoretical_power')),
                date_time=request.form.get('date_time'),
                temperature=float(request.form.get('temperature')) if request.form.get('temperature') else None
            )
            
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Input data shape: {pred_df.shape}")
            
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            return render_template('home.html', results=results[0])
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return render_template('home.html', results="Error in prediction. Please check your inputs.")

@app.route('/health')
def health_check():
    return {"status": "healthy", "message": "Wind Turbine ML API is running"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)