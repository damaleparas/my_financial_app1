import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import os

class InvestmentRiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            self.model = xgb.Booster()
            self.model.load_model('../models/investment_risk_model.bin')
            
            with open('../models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('../models/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please train the model first using model_trainer.py")
    
    def create_engineered_features(self, input_vector):
        """Create the same engineered features used during training"""
        # Convert input vector to DataFrame with proper column names
        feature_names = [
            "revenue_2021", "revenue_2022", "revenue_2023", "revenue_2024", "revenue_2025",
            "profit_2021", "profit_2022", "profit_2023", "profit_2024", "profit_2025",
            "total_debt_2021", "total_debt_2022", "total_debt_2023", "total_debt_2024", "total_debt_2025",
            "total_assets_2021", "total_assets_2022", "total_assets_2023", "total_assets_2024", "total_assets_2025",
            "net_debt_2021", "net_debt_2022", "net_debt_2023", "net_debt_2024", "net_debt_2025",
            "book_value_2021", "book_value_2022", "book_value_2023", "book_value_2024", "book_value_2025"
        ]
        
        # Create DataFrame
        df = pd.DataFrame([input_vector], columns=feature_names)
        df_eng = df.copy()
        
        years = ['2021', '2022', '2023', '2024', '2025']
        
        # Financial ratios (same as in model_trainer.py)
        for year in years:
            df_eng[f'debt_to_assets_{year}'] = df_eng[f'total_debt_{year}'] / (df_eng[f'total_assets_{year}'] + 1e-6)
            df_eng[f'profit_margin_{year}'] = df_eng[f'profit_{year}'] / (df_eng[f'revenue_{year}'] + 1e-6)
            df_eng[f'roa_{year}'] = df_eng[f'profit_{year}'] / (df_eng[f'total_assets_{year}'] + 1e-6)
            df_eng[f'book_to_assets_{year}'] = df_eng[f'book_value_{year}'] / (df_eng[f'total_assets_{year}'] + 1e-6)
        
        # Growth rates
        for metric in ['revenue', 'profit', 'total_assets', 'book_value']:
            df_eng[f'{metric}_growth'] = ((df_eng[f'{metric}_2023'] - df_eng[f'{metric}_2021']) / 
                                       (df_eng[f'{metric}_2021'] + 1e-6))
        
        # Financial health score
        df_eng['financial_health'] = (
            (df_eng['profit_2023'] - df_eng['profit_2021']) / (df_eng['profit_2021'] + 1e-6) -
            (df_eng['total_debt_2023'] - df_eng['total_debt_2021']) / (df_eng['total_debt_2021'] + 1e-6)
        )
        
        # Replace inf and nan values
        df_eng = df_eng.replace([np.inf, -np.inf], 0)
        df_eng = df_eng.fillna(0)
        
        return df_eng.values[0]  # Return as array
    
    def predict_risk(self, input_vector):
        """Predict investment risk for a company"""
        if self.model is None:
            return "Model not loaded", 0.0
        
        try:
            # Apply feature engineering (this creates 55 features from 30)
            engineered_features = self.create_engineered_features(input_vector)
            
            # Convert to numpy array and reshape
            input_array = np.array(engineered_features).reshape(1, -1)
            
            print(f"🔧 Features after engineering: {input_array.shape[1]}")
            
            # Scale features
            input_scaled = self.scaler.transform(input_array)
            
            # Create DMatrix
            dinput = xgb.DMatrix(input_scaled)
            
            # Make prediction
            prediction = self.model.predict(dinput)
            
            # Get predicted class and confidence
            if prediction.ndim > 1 and prediction.shape[1] == 3:
                pred_class = int(np.argmax(prediction[0]))
                confidence = float(np.max(prediction[0]))
            else:
                pred_class = int(prediction[0])
                confidence = 1.0
            
            # Map to labels
            risk_labels = self.label_encoder.classes_
            result = risk_labels[pred_class]
            
            return result, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Error", 0.0

def prediction_pipeline(balance_sheet_path):
    """Complete prediction pipeline"""
    from .financial_engine import get_info
    from .ml_data_extration import model_passing_for_prediction
    
    # Extract data from balance sheet
    extracted_metrics = get_info(balance_sheet_path)
    
    # Convert to feature vector
    input_vector = model_passing_for_prediction(extracted_metrics)
    
    print(f"📊 Raw features extracted: {len(input_vector)}")
    
    # Make prediction
    predictor = InvestmentRiskPredictor()
    result, confidence = predictor.predict_risk(input_vector)
    
    return result, confidence, input_vector