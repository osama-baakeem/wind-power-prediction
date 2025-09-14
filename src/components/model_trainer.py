import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Define models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                "Extra Trees": ExtraTreesRegressor(random_state=42),
            }
            
            # Hyperparameter grids for top models
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'max_depth': [3, 5, 7, 10]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7]
                },
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200, 300]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200]
                }
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                         models=models, param_grids=params)
            
            # Get the best model score from dict
            best_model_score = max([model_report[model]['test_r2_score'] for model in model_report])
            
            # Get the best model name from dict
            best_model_name = None
            best_model = None
            
            for model_name in model_report:
                if model_report[model_name]['test_r2_score'] == best_model_score:
                    best_model_name = model_name
                    best_model = model_report[model_name]['model']
                    break
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score > 0.6", sys)
            
            logging.info(f"Best found model: {best_model_name}")
            logging.info(f"Best model R2 score: {best_model_score}")

            # For XGBoost, perform additional hyperparameter tuning
            if best_model_name == "XGBRegressor":
                logging.info("Performing advanced hyperparameter tuning for XGBRegressor")
                
                param_distributions = {
                    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 
                    'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'n_estimators': [500, 750, 1000],
                    'subsample': [0.6, 0.7, 0.8, 0.9]
                }
                
                random_search = RandomizedSearchCV(
                    estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
                    param_distributions=param_distributions,
                    n_iter=20,
                    scoring='r2',
                    cv=3,
                    verbose=1,
                    random_state=42,
                    n_jobs=-1
                )
                
                random_search.fit(X_train, y_train)
                best_model = random_search.best_estimator_
                
                # Evaluate the tuned model
                predictions = best_model.predict(X_test)
                tuned_r2 = r2_score(y_test, predictions)
                tuned_rmse = np.sqrt(mean_squared_error(y_test, predictions))
                
                logging.info(f"Tuned XGBoost R2 score: {tuned_r2:.4f}")
                logging.info(f"Tuned XGBoost RMSE: {tuned_rmse:.4f}")
                
                best_model_score = tuned_r2

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score
            
        except Exception as e:
            raise CustomException(e, sys)