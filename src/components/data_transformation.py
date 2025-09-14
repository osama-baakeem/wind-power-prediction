import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from astral import LocationInfo
from astral.sun import sun
import pytz
from meteostat import Point, Hourly
import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        # Wind turbine location 
        self.location = LocationInfo("Izmir", "Turkey", "Europe/Istanbul", 38.4192, 27.1287)


    def get_deviation(self, wind_direction):
        """Calculate deviation from optimal wind directions (60° and 210°)"""
        optimal_angles = [60, 210]
        deviations = [
            min(
                abs(wind_direction - angle),
                abs(wind_direction - angle + 360),
                abs(wind_direction - angle - 360)
            ) for angle in optimal_angles
        ]
        return min(deviations)
    

    def is_day_or_night(self, dt):
        """Determine if timestamp is day (0) or night (1) based on sunrise/sunset"""
        try:
            s = sun(self.location.observer, date=dt.date())
            sunrise = s['sunrise']
            sunset = s['sunset']
            
            # Localize datetime to timezone
            if dt.tz is None:
                dt = dt.tz_localize('Europe/Istanbul')
            else:
                dt = dt.tz_convert('Europe/Istanbul')
            
            return 0 if sunrise < dt < sunset else 1  # 0=Day, 1=Night
        except Exception:
            # Fallback: simple hour-based determination
            return 0 if 6 <= dt.hour <= 18 else 1
        

    def calculate_turbine_parameters(self, df):
        """Calculate cut-in speed and rated velocity from the data"""
        try:
            # Cut-in speed: minimum wind speed where theoretical power > 0 
            # or The minimum wind speed at which the turbine starts producing power.
            cut_in_speed = df[df['Theoretical_Power_Curve (KWh)'] > 0]['Wind Speed (m/s)'].min()
            cut_in_speed = round(cut_in_speed, 1) if not pd.isna(cut_in_speed) else 3.0
            
            # Rated velocity: minimum wind speed at maximum power output
            # The wind speed at which the turbine reaches its maximum (rated) power.
            max_power = df['LV ActivePower (kW)'].max()
            rated_velocity = df[df['LV ActivePower (kW)'] == max_power]['Wind Speed (m/s)'].min()
            rated_velocity = round(rated_velocity, 1) if not pd.isna(rated_velocity) else 17.9
            
            logging.info(f"Calculated turbine parameters - Cut-in: {cut_in_speed} m/s, Rated: {rated_velocity} m/s")
            return cut_in_speed, rated_velocity
            
        except Exception as e:
            logging.warning(f"Error calculating turbine parameters: {e}. Using defaults.")
            return 3.0, 17.9  # Default values

    def add_weather_data(self, df):
        """Add temperature data using Meteostat if not present"""
        if 'Temperature (°C)' in df.columns:
            logging.info("Temperature data already present in dataset")
            return df
        
        try:
            logging.info("Fetching weather data from Meteostat...")
            
            # Define location and time period
            location = Point(38.4192, 27.1287)  # Izmir coordinates
            start = df['Date/Time'].min()
            end = df['Date/Time'].max()
            
            # Fetch hourly weather data
            data_hourly = Hourly(location, start, end)
            data_hourly = data_hourly.fetch()
            
            if not data_hourly.empty:
                # Remove timezone info and resample to 10-minute intervals
                data_hourly.index = data_hourly.index.tz_localize(None)
                data_10min = data_hourly.resample('10T').ffill().reset_index()
                
                # Prepare weather dataframe
                weather_df = data_10min[['time', 'temp']].copy()
                weather_df.rename(columns={'time': 'Date/Time', 'temp': 'Temperature (°C)'}, inplace=True)
                weather_df['Date/Time'] = pd.to_datetime(weather_df['Date/Time'])
                
                # Merge with original data
                df = pd.merge(df, weather_df, on='Date/Time', how='left')
                
                # Interpolate missing temperature values, interploate = linear estimate between known points
                df['Temperature (°C)'] = df['Temperature (°C)'].interpolate()
                
                logging.info("Weather data successfully added and interpolated")
            else:
                logging.warning("No weather data available, using synthetic temperature")
                # Create synthetic temperature based on seasonal patterns
                df['Temperature (°C)'] = (
                    15 + 10 * np.sin(2 * np.pi * (df['Date/Time'].dt.dayofyear - 80) / 365) +
                    5 * np.sin(2 * np.pi * df['Date/Time'].dt.hour / 24)
                )
                
        except Exception as e:
            logging.warning(f"Error fetching weather data: {e}. Using synthetic temperature.")
            # Fallback: synthetic temperature
            df['Temperature (°C)'] = (
                15 + 10 * np.sin(2 * np.pi * (df['Date/Time'].dt.dayofyear - 80) / 365) +
                5 * np.sin(2 * np.pi * df['Date/Time'].dt.hour / 24)
            )
        
        return df


    def add_weather_data(self, df):
        """Add temperature data using Meteostat if not present"""
        if 'Temperature (°C)' in df.columns:
            logging.info("Temperature data already present in dataset")
            return df
        
        try:
            logging.info("Fetching weather data from Meteostat...")
            
            # Define location and time period
            location = Point(38.4192, 27.1287)  # Izmir coordinates
            start = df['Date/Time'].min()
            end = df['Date/Time'].max()
            
            # Fetch hourly weather data
            data_hourly = Hourly(location, start, end)
            data_hourly = data_hourly.fetch()
            
            if not data_hourly.empty:
                # Remove timezone info and resample to 10-minute intervals
                data_hourly.index = data_hourly.index.tz_localize(None)
                data_10min = data_hourly.resample('10T').ffill().reset_index()
                
                # Prepare weather dataframe
                weather_df = data_10min[['time', 'temp']].copy()
                weather_df.rename(columns={'time': 'Date/Time', 'temp': 'Temperature (°C)'}, inplace=True)
                weather_df['Date/Time'] = pd.to_datetime(weather_df['Date/Time'])
                
                # Merge with original data
                df = pd.merge(df, weather_df, on='Date/Time', how='left')
                
                # Interpolate missing temperature values
                df['Temperature (°C)'] = df['Temperature (°C)'].interpolate()
                
                logging.info("Weather data successfully added and interpolated")
            else:
                logging.warning("No weather data available, using synthetic temperature")
                # Create synthetic temperature based on seasonal patterns
                df['Temperature (°C)'] = (
                    15 + 10 * np.sin(2 * np.pi * (df['Date/Time'].dt.dayofyear - 80) / 365) +
                    5 * np.sin(2 * np.pi * df['Date/Time'].dt.hour / 24)
                )
                
        except Exception as e:
            logging.warning(f"Error fetching weather data: {e}. Using synthetic temperature.")
            # Fallback: synthetic temperature
            df['Temperature (°C)'] = (
                15 + 10 * np.sin(2 * np.pi * (df['Date/Time'].dt.dayofyear - 80) / 365) +
                5 * np.sin(2 * np.pi * df['Date/Time'].dt.hour / 24)
            )
        
        return df



    def engineer_features(self, df):
        """Comprehensive feature engineering based on wind power domain knowledge"""
        try:
            logging.info("Starting comprehensive feature engineering...")
            
            # 1. DATETIME PREPROCESSING
            df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
            
            # 2. POWER DATA CLEANING
            # Replace negative power values with mean (from EDA: 1307.684332)
            negative_power_mean = df.loc[df['LV ActivePower (kW)'] >= 0, 'LV ActivePower (kW)'].mean()
            df.loc[df['LV ActivePower (kW)'] < 0, 'LV ActivePower (kW)'] = negative_power_mean
            
            # 3. TEMPORAL FEATURES
            df['Month'] = df['Date/Time'].dt.month
            df['Week'] = df['Date/Time'].dt.isocalendar().week
            df['Day'] = df['Date/Time'].dt.day
            df['Hour'] = df['Date/Time'].dt.hour
            df['DayOfYear'] = df['Date/Time'].dt.dayofyear
            df['DayOfWeek'] = df['Date/Time'].dt.dayofweek
            
            # 4. SEASONAL FEATURES
            seasons_dict = {
                1: 'Winter', 2: 'Winter', 3: 'Winter',
                4: 'Spring', 5: 'Spring', 6: 'Spring',
                7: 'Summer', 8: 'Summer', 9: 'Summer',
                10: 'Autumn', 11: 'Autumn', 12: 'Autumn'
            }
            df['Seasons'] = df['Month'].map(seasons_dict)
            
            # 5. CYCLICAL TIME FEATURES
            df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * (df['Month'] - 1) / 12)
            df['month_cos'] = np.cos(2 * np.pi * (df['Month'] - 1) / 12)
            
            # 6. WIND VECTOR COMPONENTS
            rad = np.deg2rad(df['Wind Direction (°)'])
            df['wind_u'] = df['Wind Speed (m/s)'] * np.cos(rad)
            df['wind_v'] = df['Wind Speed (m/s)'] * np.sin(rad)
            
            # 7. TURBULENCE & VARIABILITY FEATURES (Rolling Statistics)
            # 10-min data: window=6 → 1 hour
            df['wind_mean_1h'] = df['Wind Speed (m/s)'].rolling(window=6, min_periods=1).mean()
            df['wind_std_1h'] = df['Wind Speed (m/s)'].rolling(window=6, min_periods=1).std().fillna(0)  # shows how much wind fluctuates – a measure of turbulence.
            df['TI_1h'] = df['wind_std_1h'] / (df['wind_mean_1h'] + 1e-6)  # TI (Turbulence Intensity) = std/mean
            df['wind_max_1h'] = df['Wind Speed (m/s)'].rolling(window=6).max()
            
            # 8. POWER-POLYNOMIAL FEATURES
            # Power ∝ v³, so include v², v³ explicitly
            df['v2'] = df['Wind Speed (m/s)'] ** 2
            df['v3'] = df['Wind Speed (m/s)'] ** 3
            
            # 9. TURBINE REGIME CLASSIFICATION
            # Classify turbine operation into three regimes based on wind speed:
            # 1. 'below_cut_in' → wind too low, turbine produces no power, below cut-in speed
            # 2. 'partial'      → turbine generating power but below rated capacity
            # 3. 'rated'        → turbine at maximum (rated) power
            # 'pd.cut' bins the wind speed into these categories and assigns labels
            cut_in_speed, rated_velocity = self.calculate_turbine_parameters(df)
            df['regime'] = pd.cut(
                df['Wind Speed (m/s)'],
                bins=[-1, cut_in_speed, rated_velocity, 1e9],
                labels=['below_cut_in', 'partial', 'rated']
            )
         
            # 10. DAY/NIGHT FEATURE
            df['Day/Night'] = df['Date/Time'].apply(self.is_day_or_night)
            
            # 11. ADD WEATHER DATA
            df = self.add_weather_data(df)
            
            # 12. WIND DIRECTION OPTIMIZATION
            # Effective Theoretical Power based on wind direction deviation
            ## df['Wind_Direction_Deviation'] / 360 → fraction of the full circle (0–1).
            ## * 100 → convert fraction to a percentage of loss due to misalignment.
            ## 100 - ... → gives the percentage of power actually captured.
            ## Multiply by Theoretical_Power_Curve (KWh) → scales theoretical power by alignment efficiency.
            ## / 100 → convert back to the original kWh unit.
            ## Example:
            ## Theoretical power = 1000 kWh, deviation = 36°
            ## Loss percentage = 36 / 360 * 100 = 10%
            ## Effective power = 1000 * (100 - 10) / 100 = 900 kWh
            df['Wind_Direction_Deviation'] = df['Wind Direction (°)'].apply(self.get_deviation)
            df['Effective Theoretical Power(kWh)'] = (
                (100 - (df['Wind_Direction_Deviation'] / 360) * 100) * 
                df['Theoretical_Power_Curve (KWh)'] / 100
            )
            
            # 13. ADVANCED WIND POWER FEATURES
            # Wind Power Density: kinetic energy available in the wind per unit area (W/m²)
            # Formula: 0.5 * air_density * wind_speed^3
            # 1.225 kg/m³ is standard air density at sea level
            # Wind speed is in m/s, and cubed because kinetic energy increases with the cube of velocity
            df['Wind Power Density'] = 0.5 * 1.225 * (df['Wind Speed (m/s)'] ** 3)
            
            # Air density adjustment based on temperature
            # Air density decreases with higher temperature (ρ ∝ 1/T)
            # - 1.225 kg/m³: standard air density at 15°C
            # - 288.15 K: standard temperature reference (15°C)
            # Formula: ρ = 1.225 * (288.15 / (T + 273.15))
            # Temperature Adjusted Power scales Wind Power Density by actual air density
            if 'Temperature (°C)' in df.columns:
                df['Air Density'] = 1.225 * (288.15 / (df['Temperature (°C)'] + 273.15))
                df['Temperature Adjusted Power'] = df['Wind Power Density'] * (df['Air Density'] / 1.225)
            
            # Wind direction efficiency: accounts for misalignment of wind with turbine optimal direction
            # - Wind_Direction_Deviation: deviation from optimal angles (60° or 210°)
            # - Efficiency = cos(deviation in radians)
            #   cos(0°) = 1 → perfect alignment, cos(90°) = 0 → perpendicular → zero efficiency
            df['Wind Direction Efficiency'] = np.cos(np.radians(df['Wind_Direction_Deviation']))
            
            # 14. POWER CURVE ANALYSIS
            # Power Curve Efficiency: measures actual power output relative to theoretical maximum
            # If theoretical power > 0, efficiency = actual / theoretical
            # If theoretical power = 0 (no wind), set efficiency = 0
            df['Power Curve Efficiency'] = np.where(
                df['Theoretical_Power_Curve (KWh)'] > 0,
                df['LV ActivePower (kW)'] / df['Theoretical_Power_Curve (KWh)'],
                0
            )
            
            # 15. WIND SPEED CATEGORIZATION
            partial_mask = df['regime'] == 'partial'
            df.loc[partial_mask, 'Wind Speed Category'] = pd.cut(
                df.loc[partial_mask, 'Wind Speed (m/s)'],
                bins=[cut_in_speed, 7, 12, rated_velocity],
                labels=['Low', 'Medium', 'High']
            )

            
            # 16. ADDITIONAL WIND DIRECTION FEATURES
            df['Wind Direction Sin'] = np.sin(rad)
            df['Wind Direction Cos'] = np.cos(rad)
            
            # 17. CLEAN UP MISSING VALUES FROM ROLLING STATISTICS
            df.dropna(subset=['wind_max_1h'], inplace=True)
            
            logging.info(f"Feature engineering completed. Final dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            raise Exception(f"Error in feature engineering: {str(e)}")




    def get_preprocessor(self, df):
        """Create preprocessing pipeline based on available features"""
        try:
            # Define feature categories
            numerical_features = []
            categorical_features = []
            
            # Base numerical features (always present)
            base_numerical = [
                'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)',
                'Month', 'Week', 'Day', 'Hour', 'DayOfYear', 'DayOfWeek',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'wind_u', 'wind_v', 'wind_mean_1h', 'wind_std_1h', 'TI_1h', 'wind_max_1h',
                'v2', 'v3', 'Day/Night', 'Wind_Direction_Deviation',
                'Effective Theoretical Power(kWh)', 'Wind Power Density',
                'Wind Direction Efficiency', 'Power Curve Efficiency',
                'Wind Direction Sin', 'Wind Direction Cos'
            ]
            
            # Optional numerical features (temperature-related)
            optional_numerical = ['Temperature (°C)', 'Air Density', 'Temperature Adjusted Power']
            
            # Categorical features
            categorical_features = ['Seasons', 'regime', 'Wind Speed Category']
            
            # Select only available features
            available_columns = df.columns.tolist()
            numerical_features = [col for col in base_numerical + optional_numerical 
                                if col in available_columns]
            categorical_features = [col for col in categorical_features 
                                  if col in available_columns]
            
            logging.info(f"Preprocessing pipeline - Numerical features: {len(numerical_features)}")
            logging.info(f"Preprocessing pipeline - Categorical features: {len(categorical_features)}")
            
            # Create preprocessing pipelines
            numerical_pipeline = Pipeline(
                [
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
                    ('scaler', StandardScaler(with_mean=False))
            ]
            )
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                steps=[
                    ('num', numerical_pipeline, numerical_features),
                    ('cat', categorical_pipeline, categorical_features)
            ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise Exception(f"Error creating preprocessor: {str(e)}")
    
    def initiate_data_transformation(self, train_path, test_path):
        """Main method to execute the complete data transformation pipeline"""
        try:
            logging.info("Starting data transformation pipeline...")
            
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Original train data shape: {train_df.shape}")
            logging.info(f"Original test data shape: {test_df.shape}")
            
            # Apply feature engineering
            train_df = self.engineer_features(train_df.copy())
            test_df = self.engineer_features(test_df.copy())
            
            # Define target column
            target_column = 'LV ActivePower (kW)'
            
            # Drop unnecessary columns
            columns_to_drop = ['Date/Time']
            
            # Separate features and target
            X_train = train_df.drop(columns=columns_to_drop + [target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=columns_to_drop + [target_column])
            y_test = test_df[target_column]
            
            # Get preprocessor
            preprocessor = self.get_preprocessor(X_train)
            
            # Apply preprocessing
            logging.info("Applying preprocessing transformations...")
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            
            # Combine features and target for compatibility
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]
            
            # Save preprocessor
            self._save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            
            logging.info("Data transformation completed successfully!")
            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise Exception(f"Error in data transformation pipeline: {str(e)}")
    
    def _save_object(self, file_path, obj):
        """Save preprocessing object to file"""
        try:
            import pickle
            
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
                
        except Exception as e:
            raise Exception(f"Error saving object: {str(e)}")



# Example usage and testing
if __name__ == "__main__":
    # Initialize transformation pipeline
    transformation = DataTransformation()
    
    # Example of how to use the pipeline
    try:
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path="path/to/train.csv",
            test_path="path/to/test.csv"
        )
        print("Data transformation completed successfully!")
        print(f"Train array shape: {train_arr.shape}")
        print(f"Test array shape: {test_arr.shape}")
        print(f"Preprocessor saved at: {preprocessor_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


























