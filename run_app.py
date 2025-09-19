#!/usr/bin/env python3
"""
Startup script for ForeTel.AI application
This script handles initialization, dependency checking, and app startup
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'scikit-learn',
        'joblib', 'matplotlib', 'xgboost', 'lightgbm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Run: pip install -r Requirements.txt")
        return False
    
    return True

def setup_directories():
    """Create necessary directories."""
    directories = ['models', 'data', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directory ready: {directory}")

def check_data_files():
    """Check if data files exist, generate if missing."""
    data_files = [
        'telecom_dataset.csv',
        'telecom_dataset_generated.csv',
        'telecom_dataset_preprocessed.csv'
    ]
    
    existing_files = [f for f in data_files if os.path.exists(f)]
    
    if not existing_files:
        logger.warning("No data files found. Generating sample dataset...")
        try:
            from dataset_generator import generate_telecom_dataset
            df = generate_telecom_dataset(1000)  # Generate smaller dataset for demo
            df.to_csv('telecom_dataset_generated.csv', index=False)
            logger.info("Sample dataset generated: telecom_dataset_generated.csv")
        except Exception as e:
            logger.error(f"Failed to generate dataset: {e}")
            return False
    else:
        logger.info(f"Data files found: {', '.join(existing_files)}")
    
    return True

def check_model_files():
    """Check if model files exist."""
    model_files = [
        'models/enhanced_churn_model.pkl',
        'models/revenue_forecasting.pkl',
        'models/scaler.pkl',
        'models/train_columns.pkl',
        'models/imputer.pkl'
    ]
    
    existing_models = [f for f in model_files if os.path.exists(f)]
    missing_models = [f for f in model_files if not os.path.exists(f)]
    
    if existing_models:
        logger.info(f"Model files found: {len(existing_models)}/{len(model_files)}")
    
    if missing_models:
        logger.warning(f"Missing model files: {len(missing_models)}")
        logger.info("Some features may not work without trained models.")
        logger.info("Run training scripts to generate missing models.")

def run_streamlit_app():
    """Run the Streamlit application."""
    try:
        logger.info("Starting ForeTel.AI application...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit app: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        return True

def main():
    """Main startup function."""
    logger.info("=" * 50)
    logger.info("ForeTel.AI - Telecom Analytics Platform")
    logger.info("=" * 50)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        logger.error("Please install missing dependencies and try again")
        sys.exit(1)
    
    # Setup environment
    setup_directories()
    
    if not check_data_files():
        logger.error("Failed to setup data files")
        sys.exit(1)
    
    check_model_files()
    
    # Start application
    logger.info("All checks passed. Starting application...")
    run_streamlit_app()

if __name__ == "__main__":
    main()
