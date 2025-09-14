import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            logging.info("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("After Loading")
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 wind_speed: float,
                 wind_direction: float,
                 theoretical_power: float,
                 date_time: str,
                 temperature: float = None):
        
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.theoretical_power = theoretical_power
        self.date_time = date_time
        self.temperature = temperature

    def get_deviation(self, wind_direction):
        """Calculate deviation from optimal wind directions"""
        optimal_angles = [60, 210]
        deviations = [
            min(
                abs(wind_direction - angle),
                abs(wind_direction - angle + 360),
                abs(wind_direction - angle - 360)
            ) for angle in optimal_angles
        ]
        return min(deviations)

    def get_data_as_data_frame(self):
        try:
            # Parse datetime
            dt = pd.to_datetime(self.date_time)
            
            # Extract temporal features
            week = dt.isocalendar().week
            month = dt.month
            hour = dt.hour
            day_of_year = dt.dayofyear
            day_of_week = dt.dayofweek
            
            # Season mapping
            season_mapping = {
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Autumn", 10: "Autumn", 11: "Autumn"
            }
            season = season_mapping[month]
            
            # Day/Night (simplified - based on hour)
            day_night = 0 if 6 <= hour <= 18 else 1
            
            # Calculate engineered features
            effective_theoretical_power = (
                100 - ((self.get_deviation(self.wind_direction) / 360) * 100)
            ) * self.theoretical_power / 100
            
            wind_power_density = 0.5 * 1.225 * (self.wind_speed ** 3)
            wind_direction_efficiency = np.cos(np.radians(self.get_deviation(self.wind_direction)))
            power_curve_efficiency = 1.0  # Default value
            
            # Wind speed category
            cut_in_speed = 3.0
            rated_velocity = 17.9
            
            if self.wind_speed < cut_in_speed:
                wind_speed_category = "Below Cut-in"
            elif self.wind_speed < 7:
                wind_speed_category = "Low"
            elif self.wind_speed < 12:
                wind_speed_category = "Medium"
            elif self.wind_speed < rated_velocity:
                wind_speed_category = "High"
            else:
                wind_speed_category = "Above Rated"
            
            # Rolling statistics (simplified)
            wind_speed_rolling_mean = self.wind_speed
            wind_speed_rolling_std = 0.1  # Default small value
            turbulence_intensity = wind_speed_rolling_std / wind_speed_rolling_mean
            
            # Cyclic features
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            wind_direction_sin = np.sin(np.radians(self.wind_direction))
            wind_direction_cos = np.cos(np.radians(self.wind_direction))
            
            custom_data_input_dict = {
                "Wind Speed (m/s)": [self.wind_speed],
                "Theoretical_Power_Curve (KWh)": [self.theoretical_power],
                "Wind Direction (°)": [self.wind_direction],
                "Week": [week],
                "Month": [month],
                "Hour": [hour],
                "DayOfYear": [day_of_year],
                "DayOfWeek": [day_of_week],
                "Day/Night": [day_night],
                "Season": [season],
                "Effective Theoretical Power(kWh)": [effective_theoretical_power],
                "Wind Power Density": [wind_power_density],
                "Wind Direction Efficiency": [wind_direction_efficiency],
                "Power Curve Efficiency": [power_curve_efficiency],
                "Wind Speed Category": [wind_speed_category],
                "Wind Speed Rolling Mean": [wind_speed_rolling_mean],
                "Wind Speed Rolling Std": [wind_speed_rolling_std],
                "Turbulence Intensity": [turbulence_intensity],
                "Month_Sin": [month_sin],
                "Month_Cos": [month_cos],
                "Hour_Sin": [hour_sin],
                "Hour_Cos": [hour_cos],
                "Wind Direction Sin": [wind_direction_sin],
                "Wind Direction Cos": [wind_direction_cos]
            }
            
            # Add temperature features if provided
            if self.temperature is not None:
                air_density = 1.225 * (288.15 / (self.temperature + 273.15))
                temperature_adjusted_power = wind_power_density * (air_density / 1.225)
                
                custom_data_input_dict.update({
                    "Temperature (°C)": [self.temperature],
                    "Air Density": [air_density],
                    "Temperature Adjusted Power": [temperature_adjusted_power]
                })

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)