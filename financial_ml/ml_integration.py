# financial_ml/ml_integration.py

import os
import pandas as pd
from django.conf import settings # Import Django settings

# Import your existing ML modules
from MultipleFiles.model_trainer import InvestmentRiskTrainer, generate_sample_data
from MultipleFiles.predictor import prediction_pipeline, InvestmentRiskPredictor
from MultipleFiles.financial_engine import get_info
from MultipleFiles.ml_data_extration import model_passing_for_prediction
from MultipleFiles.model_trainer import InvestmentRiskTrainer, generate_sample_data
# Ensure models directory exists and is accessible
# In a production environment, you might store these in a more persistent storage
# or ensure your deployment process handles model loading paths correctly.
MODEL_DIR = os.path.join(settings.BASE_DIR, 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class MLService:
    def __init__(self):
        # Initialize trainer and predictor.
        # The predictor will attempt to load models on initialization.
        # Ensure your predictor.py and model_trainer.py are updated to use
        # os.path.join(settings.BASE_DIR, 'models', ...) for model paths.
        self.trainer = InvestmentRiskTrainer()
        self.predictor = InvestmentRiskPredictor()

    def train_model_sample_data(self):
        """Trains the model using generated sample data."""
        print("📊 Training model with sample data...")
        sample_df = generate_sample_data(1000)
        self.trainer.train_model(df=sample_df)
        print("✅ Model training completed!")

    def train_model_custom_data(self, csv_file_path):
        """Trains the model using custom CSV data."""
        print(f"\n📊 Training model with {csv_file_path}...")
        self.trainer.train_model(df_path=csv_file_path)
        print("✅ Model training completed!")

    def make_prediction_from_balance_sheet(self, balance_sheet_csv_path):
        """Makes a prediction from a balance sheet CSV."""
        print(f"\n🔮 Making prediction for {balance_sheet_csv_path}...")
        try:
            # prediction_pipeline is from MultipleFiles.predictor
            result, confidence, input_vector = prediction_pipeline(balance_sheet_csv_path)
            return {
                "risk_category": result,
                "confidence": f"{confidence:.3f}",
                "input_vector_length": len(input_vector)
            }
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

    def test_prediction_with_sample_data(self):
        """Tests prediction with a hardcoded sample input vector."""
        print("\n🧪 Testing with sample data...")
        test_input = [484, 613, 702, 0, 0, 23, 35, 42, 0, 0,
                      104, 123, 108, 0, 0, 452, 514, 523, 0, 0,
                      104, 123, 108, 0, 0, 215, 252, 294, 0, 0]
        # predict_risk is a method of InvestmentRiskPredictor from MultipleFiles.predictor
        result, confidence = self.predictor.predict_risk(test_input)
        return {
            "risk_category": result,
            "confidence": f"{confidence:.3f}"
        }

# Initialize the MLService. This will load the models when the Django app starts.
ml_service = MLService()
