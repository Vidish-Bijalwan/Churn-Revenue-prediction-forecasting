#!/usr/bin/env python3
"""
Test script to validate trained models and their predictions.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import os

def test_churn_model():
    """Test the churn prediction model."""
    print("🔍 Testing Churn Prediction Model...")
    
    try:
        # Load the trained model
        churn_model = joblib.load('models/enhanced_churn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        train_columns = joblib.load('models/train_columns.pkl')
        
        print("✅ Churn model files loaded successfully")
        
        # Create test data
        test_data = pd.DataFrame({
            'Age': [35, 45, 28, 55],
            'TenureMonths': [24, 48, 12, 60],
            'MonthlyCharges': [75.0, 85.0, 65.0, 95.0],
            'TotalCharges': [1800.0, 4080.0, 780.0, 5700.0],
            'DataUsageGB': [25.0, 35.0, 15.0, 45.0],
            'MinutesUsed': [500, 700, 300, 800],
            'TextsSent': [150, 200, 100, 250],
            'CustomerServiceCalls': [2, 1, 3, 0],
            'Gender_Male': [1, 0, 1, 0],
            'Region_North': [1, 0, 0, 0],
            'Region_South': [0, 1, 0, 0],
            'Region_East': [0, 0, 1, 0],
            'Region_West': [0, 0, 0, 1],
            'SubscriptionType_Premium': [0, 1, 0, 1],
            'SubscriptionType_Standard': [1, 0, 1, 0],
            'ContractType_One Year': [1, 1, 0, 1],
            'ContractType_Two Year': [0, 0, 0, 0],
            'PaymentMethod_Credit Card': [1, 0, 1, 0],
            'PaymentMethod_Electronic Check': [0, 1, 0, 0],
            'PaymentMethod_Mailed Check': [0, 0, 0, 1],
            'InternationalPlan': [0, 1, 0, 1],
            'VoiceMailPlan': [1, 0, 1, 0]
        })
        
        # Align columns with training data
        for col in train_columns:
            if col not in test_data.columns:
                test_data[col] = 0
        test_data = test_data[train_columns]
        
        # Scale the data
        test_data_scaled = scaler.transform(test_data)
        
        # Make predictions
        predictions = churn_model.predict(test_data_scaled)
        probabilities = churn_model.predict_proba(test_data_scaled)
        
        print(f"✅ Churn predictions: {predictions}")
        print(f"✅ Churn probabilities: {probabilities[:, 1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Churn model test failed: {e}")
        return False

def test_revenue_model():
    """Test the revenue forecasting model."""
    print("\n💰 Testing Revenue Forecasting Model...")
    
    try:
        # Load the trained model
        revenue_model = joblib.load('models/revenue_forecasting.pkl')
        
        print("✅ Revenue model loaded successfully")
        
        # Create test data (same structure as training)
        test_data = pd.DataFrame({
            'Age': [35, 45, 28, 55],
            'TenureMonths': [24, 48, 12, 60],
            'MonthlyCharges': [75.0, 85.0, 65.0, 95.0],
            'DataUsageGB': [25.0, 35.0, 15.0, 45.0],
            'MinutesUsed': [500, 700, 300, 800],
            'TextsSent': [150, 200, 100, 250],
            'CustomerServiceCalls': [2, 1, 3, 0],
            'Gender_Male': [1, 0, 1, 0],
            'Region_North': [1, 0, 0, 0],
            'Region_South': [0, 1, 0, 0],
            'Region_East': [0, 0, 1, 0],
            'Region_West': [0, 0, 0, 1],
            'SubscriptionType_Premium': [0, 1, 0, 1],
            'SubscriptionType_Standard': [1, 0, 1, 0],
            'ContractType_One Year': [1, 1, 0, 1],
            'ContractType_Two Year': [0, 0, 0, 0],
            'PaymentMethod_Credit Card': [1, 0, 1, 0],
            'PaymentMethod_Electronic Check': [0, 1, 0, 0],
            'PaymentMethod_Mailed Check': [0, 0, 0, 1],
            'InternationalPlan': [0, 1, 0, 1],
            'VoiceMailPlan': [1, 0, 1, 0]
        })
        
        # Make predictions
        predictions = revenue_model.predict(test_data)
        
        print(f"✅ Revenue predictions: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Revenue model test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\n📊 Testing Data Loading...")
    
    try:
        # Check if dataset exists
        dataset_files = [
            'telecom_dataset_generated.csv',
            'telecom_dataset.csv',
            'telecom_dataset_preprocessed.csv'
        ]
        
        dataset_found = False
        for file in dataset_files:
            if Path(file).exists():
                df = pd.read_csv(file)
                print(f"✅ Dataset loaded: {file} ({len(df)} rows, {len(df.columns)} columns)")
                dataset_found = True
                break
        
        if not dataset_found:
            print("❌ No dataset found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ForeTel.AI Model Testing Suite")
    print("=" * 50)
    
    # Change to project directory
    os.chdir('/home/zerosirus/Desktop/major projects/churn_prediction_revenue_forecasting/Churn-And-Revenue-Forecaster')
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Churn Model", test_churn_model),
        ("Revenue Model", test_revenue_model)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Models are ready for use.")
    else:
        print("\n⚠️  Some tests failed. Please check the models.")
    
    return all_passed

if __name__ == "__main__":
    main()
