from src.pipelines.predict_pipeline import PredictPipeline, CustomData

if __name__ == '__main__':
    cd = CustomData(wind_speed=5.0, wind_direction=60, theoretical_power=100.0, date_time='2025-09-14 12:00:00', temperature=20.0)
    df = cd.get_data_as_data_frame()
    print('Sample df shape:', df.shape)

    pp = PredictPipeline()
    try:
        preds = pp.predict(df)
        print('Prediction:', preds)
    except Exception as e:
        print('Predict error:', e)
