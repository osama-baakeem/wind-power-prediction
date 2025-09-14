# Wind Turbine Power Prediction - End-to-End ML Project

## Overview
This project implements an industry-standard machine learning system to predict wind turbine power output using SCADA data. It demonstrates advanced feature engineering, MLOps practices, and multiple deployment strategies.

## Features
- **Advanced Feature Engineering**: Temporal features, weather integration, wind-specific calculations
- **Multiple ML Models**: XGBoost, CatBoost, Random Forest, and more with automated selection
- **MLOps Pipeline**: Modular code structure, logging, exception handling, CI/CD
- **Web Interface**: Flask application for real-time predictions
- **Cloud Deployment**: Support for AWS and Azure deployment
- **Containerization**: Docker support for consistent deployments

## Project Structure
```
wind_turbine_ml/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
├── templates/
├── artifacts/
├── app.py
├── requirements.txt
├── setup.py
├── Dockerfile
└── .github/workflows/
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd wind_turbine_ml
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```python
from src.pipeline.train_pipeline import run_training_pipeline

# Train the model with your dataset
score = run_training_pipeline("path/to/your/wind_turbine_data.csv")
```

### Running the Web Application
```bash
python app.py
```
Visit `http://localhost:5000` to use the prediction interface.

### Making Predictions Programmatically
```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create input data
data = CustomData(
    wind_speed=7.5,
    wind_direction=45.0,
    theoretical_power=1500.0,
    date_time="2024-01-15 14:30:00",
    temperature=15.0
)

# Make prediction
pipeline = PredictPipeline()
prediction = pipeline.predict(data.get_data_as_data_frame())
print(f"Predicted Power: {prediction[0]} kW")
```

## Model Performance
- **Best Model**: XGBoost with advanced hyperparameter tuning
- **Expected Performance**: R² > 0.98, RMSE < 0.15
- **Features**: 20+ engineered features including weather integration

## Deployment

### Docker Deployment
```bash
docker build -t wind-turbine-ml .
docker run -p 5000:5000 wind-turbine-ml
```

### AWS Deployment
The project includes GitHub Actions workflows for automated deployment to:
- AWS Elastic Beanstalk
- AWS ECS with ECR
- AWS Lambda (serverless)

### Azure Deployment
- Azure Web Apps
- Azure Container Instances
- Azure ML Studio

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License
This project is licensed under the MIT License.
"""