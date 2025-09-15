import sys
import os
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
            
            # Try transforming features; if some columns are missing (common when running
            # with a reduced set of engineered features), add them with default values and retry.
            try:
                data_scaled = preprocessor.transform(features)
            except Exception as e:
                msg = str(e)
                # Look for a pandas-like message containing the missing columns set
                if 'columns are missing' in msg:
                    try:
                        import re
                        m = re.search(r"columns are missing: \{([^}]*)\}", msg)
                        if m:
                            cols_str = m.group(1)
                            # split and clean column names
                            missing_cols = [c.strip().strip("'\"") for c in cols_str.split(',') if c.strip()]
                            added = []
                            for c in missing_cols:
                                if c not in features.columns:
                                    features[c] = 0
                                    added.append(c)
                            logging.warning(f"Added missing columns with default 0: {added}")
                            data_scaled = preprocessor.transform(features)
                        else:
                            raise
                    except Exception:
                        # If anything goes wrong during the recovery attempt, re-raise original
                        raise
                else:
                    raise

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
            # Compute additional features using the same logic as training
            month_sin_l = np.sin(2 * np.pi * (month - 1) / 12)
            month_cos_l = np.cos(2 * np.pi * (month - 1) / 12)
            hour_sin_l = np.sin(2 * np.pi * hour / 24)
            hour_cos_l = np.cos(2 * np.pi * hour / 24)

            rad = np.deg2rad(self.wind_direction)
            wind_u = self.wind_speed * np.cos(rad)
            wind_v = self.wind_speed * np.sin(rad)

            # Rolling statistics defaults for single-row input (training used rolling with min_periods=1)
            wind_mean_1h = self.wind_speed
            # Heuristic estimate for turbulence intensity (TI) depending on wind speed.
            # These values are chosen as reasonable defaults and can be tuned per site.
            if wind_mean_1h <= 0:
                TI_1h = 0.0
            elif wind_mean_1h <= 3.0:
                TI_1h = 0.20
            elif wind_mean_1h <= 7.0:
                TI_1h = 0.15
            elif wind_mean_1h <= 12.0:
                TI_1h = 0.10
            elif wind_mean_1h <= 17.9:
                TI_1h = 0.08
            else:
                TI_1h = 0.06

            # Estimate standard deviation from TI * mean
            wind_std_1h = float(TI_1h * wind_mean_1h)
            # Estimate 1-hour max speed as mean + 2*sigma (approx)
            wind_max_1h = float(wind_mean_1h + 2.0 * wind_std_1h)

            v2 = self.wind_speed ** 2
            v3 = self.wind_speed ** 3

            day = dt.day
            wind_dir_dev = self.get_deviation(self.wind_direction)

            # Regime classification using default turbine parameters (same defaults as training)
            cut_in_speed = 3.0
            rated_velocity = 17.9
            if self.wind_speed <= cut_in_speed:
                regime = 'below_cut_in'
            elif self.wind_speed < rated_velocity:
                regime = 'partial'
            else:
                regime = 'rated'

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
                "Seasons": [season],
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

            # Add lower-case and training-expected feature names (duplicates are harmless)
            custom_data_input_dict.update({
                'month_sin': [month_sin_l],
                'month_cos': [month_cos_l],
                'hour_sin': [hour_sin_l],
                'hour_cos': [hour_cos_l],
                'wind_u': [wind_u],
                'wind_v': [wind_v],
                'wind_mean_1h': [wind_mean_1h],
                'wind_std_1h': [wind_std_1h],
                'TI_1h': [TI_1h],
                'wind_max_1h': [wind_max_1h],
                'v2': [v2],
                'v3': [v3],
                'Day': [day],
                'Seasons': [season],
                'Wind_Direction_Deviation': [wind_dir_dev],
                'regime': [regime]
            })
            
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