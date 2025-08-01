import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import os

class InvestmentRiskTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            "revenue_2021", "revenue_2022", "revenue_2023", "revenue_2024", "revenue_2025",
            "profit_2021", "profit_2022", "profit_2023", "profit_2024", "profit_2025",
            "total_debt_2021", "total_debt_2022", "total_debt_2023", "total_debt_2024", "total_debt_2025",
            "total_assets_2021", "total_assets_2022", "total_assets_2023", "total_assets_2024", "total_assets_2025",
            "net_debt_2021", "net_debt_2022", "net_debt_2023", "net_debt_2024", "net_debt_2025",
            "book_value_2021", "book_value_2022", "book_value_2023", "book_value_2024", "book_value_2025"
        ]
    
    def create_engineered_features(self, df):
        """Create financial ratios and engineered features"""
        df_eng = df.copy()
        
        years = ['2021', '2022', '2023', '2024', '2025']
        
        for year in years:
            # Financial ratios
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
        
        return df_eng
    
    def create_target_labels(self, df):
        """Create risk labels based on financial health"""
        scores = []
        
        for idx, row in df.iterrows():
            score = 0
            
            # Revenue growth
            revenue_growth = (row['revenue_2023'] - row['revenue_2021']) / (row['revenue_2021'] + 1e-6)
            if revenue_growth > 0.2: score += 2
            elif revenue_growth > 0: score += 1
            
            # Profitability
            avg_profit_margin = np.mean([row[f'profit_margin_{year}'] for year in ['2021', '2022', '2023']])
            if avg_profit_margin > 0.15: score += 2
            elif avg_profit_margin > 0.05: score += 1
            
            # Debt management
            avg_debt_ratio = np.mean([row[f'debt_to_assets_{year}'] for year in ['2021', '2022', '2023']])
            if avg_debt_ratio < 0.3: score += 2
            elif avg_debt_ratio < 0.6: score += 1
            
            # ROA
            avg_roa = np.mean([row[f'roa_{year}'] for year in ['2021', '2022', '2023']])
            if avg_roa > 0.1: score += 2
            elif avg_roa > 0.05: score += 1
            
            # Financial health trend
            if row['financial_health'] > 0: score += 1
            
            scores.append(score)
        
        # Convert to risk categories
        labels = []
        for score in scores:
            if score >= 6: labels.append('Good')
            elif score >= 3: labels.append('Moderate')
            else: labels.append('Risky')
        
        return labels
    
    def train_model(self, df_path=None, df=None):
        """Train the XGBoost model"""
        if df is None:
            df = pd.read_csv(df_path)
        
        # Create engineered features
        df_eng = self.create_engineered_features(df)
        
        # Create target labels
        if 'risk_category' not in df_eng.columns:
            df_eng['risk_category'] = self.create_target_labels(df_eng)
        
        # Prepare features and target
        X = df_eng.drop(['Company_Ticker', 'risk_category'], axis=1, errors='ignore')
        y = df_eng['risk_category']
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sample_weights = np.array([class_weights[label] for label in y_train])
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, weight=sample_weights)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)
        
        # XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Evaluate
        y_pred = self.model.predict(dtest)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        target_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=target_names))
        
        # Save model
        os.makedirs('../models', exist_ok=True)
        self.model.save_model('../models/investment_risk_model.bin')
        
        # Save preprocessing objects
        import pickle
        with open('../models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('../models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print("\nModel and preprocessing objects saved!")
        return self.model

def generate_sample_data(n_samples=1000):
    """Generate sample training data"""
    np.random.seed(42)
    
    sample_data = {
        'Company_Ticker': [f'COMP_{i:03d}' for i in range(n_samples)]
    }
    
    feature_names = [
        "revenue_2021", "revenue_2022", "revenue_2023", "revenue_2024", "revenue_2025",
        "profit_2021", "profit_2022", "profit_2023", "profit_2024", "profit_2025",
        "total_debt_2021", "total_debt_2022", "total_debt_2023", "total_debt_2024", "total_debt_2025",
        "total_assets_2021", "total_assets_2022", "total_assets_2023", "total_assets_2024", "total_assets_2025",
        "net_debt_2021", "net_debt_2022", "net_debt_2023", "net_debt_2024", "net_debt_2025",
        "book_value_2021", "book_value_2022", "book_value_2023", "book_value_2024", "book_value_2025"
    ]
    
    for feature in feature_names:
        if 'revenue' in feature:
            sample_data[feature] = np.random.lognormal(6, 0.5, n_samples)
        elif 'profit' in feature:
            sample_data[feature] = np.random.lognormal(4, 0.8, n_samples)
        elif 'debt' in feature:
            sample_data[feature] = np.random.lognormal(5, 0.6, n_samples)
        elif 'assets' in feature:
            sample_data[feature] = np.random.lognormal(7, 0.4, n_samples)
        elif 'book_value' in feature:
            sample_data[feature] = np.random.lognormal(5.5, 0.5, n_samples)
    
    return pd.DataFrame(sample_data)