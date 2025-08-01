import sys
import os
import pandas as pd
from model_trainer import InvestmentRiskTrainer, generate_sample_data
from predictor import prediction_pipeline

def main():
    print("🏦 Financial Risk Prediction System")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Train new model with sample data")
        print("2. Train model with custom data")
        print("3. Make prediction from balance sheet")
        print("4. Test prediction with sample data")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\n📊 Training model with sample data...")
            
            # Generate sample data
            sample_df = generate_sample_data(1000)
            
            # Train model
            trainer = InvestmentRiskTrainer()
            trainer.train_model(df=sample_df)
            
            print("✅ Model training completed!")
        
        elif choice == '2':
            csv_path = input("Enter path to your training CSV file: ").strip()
            
            if not os.path.exists(csv_path):
                print("❌ File not found!")
                continue
            
            print(f"\n📊 Training model with {csv_path}...")
            
            trainer = InvestmentRiskTrainer()
            trainer.train_model(df_path=csv_path)
            
            print("✅ Model training completed!")
        
        elif choice == '3':
            balance_sheet_path = input("Enter path to balance sheet CSV: ").strip()
            
            if not os.path.exists(balance_sheet_path):
                print("❌ File not found!")
                continue
            
            print(f"\n🔮 Making prediction for {balance_sheet_path}...")
            
            try:
                result, confidence, input_vector = prediction_pipeline(balance_sheet_path)
                
                print(f"\n📈 Prediction Results:")
                print(f"   Risk Category: {result}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Input Vector Length: {len(input_vector)}")
                
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
        
        elif choice == '4':
            print("\n🧪 Testing with sample data...")
            
            # Sample input vector (your example)
            test_input = [484, 613, 702, 0, 0, 23, 35, 42, 0, 0, 
                         104, 123, 108, 0, 0, 452, 514, 523, 0, 0, 
                         104, 123, 108, 0, 0, 215, 252, 294, 0, 0]
            
            from predictor import InvestmentRiskPredictor
            predictor = InvestmentRiskPredictor()
            result, confidence = predictor.predict_risk(test_input)
            
            print(f"\n📈 Test Prediction Results:")
            print(f"   Risk Category: {result}")
            print(f"   Confidence: {confidence:.3f}")
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()