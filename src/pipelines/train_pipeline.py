import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training_pipeline(data_path: str):
    try:
        # Data Ingestion
        logging.info("Starting training pipeline")
        
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(data_path)
        
        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        # Model Training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        logging.info(f"Training pipeline completed with R2 score: {r2_score}")
        return r2_score
        
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage
    data_file_path = "path/to/your/wind_turbine_data.csv"
    score = run_training_pipeline(data_file_path)
    print(f"Model training completed with R2 score: {score}")