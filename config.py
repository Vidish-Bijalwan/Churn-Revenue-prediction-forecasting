# Configuration settings for ForeTel.AI

# Custom CSS for improved UI/UX
custom_css = """
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #1F4529;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        color: white;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        margin-top: 0;
        color: #E0E0E0;
    }
    .metric-card-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .welcome-message {
        background-color: #2A603B;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 2rem;
        color: white;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 0.25rem;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        .metric-card {
            margin-bottom: 0.5rem;
        }
    }
</style>
"""

# Footer content
footer = """
<div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #f1f1f1; padding: 10px; text-align: center;">
    <p>© 2023 ForeTel.AI. All rights reserved. | <a href="#" onclick="alert('Privacy Policy: We respect your privacy and are committed to protecting your personal data.')">Privacy Policy</a></p>
</div>
"""

# Application Configuration
import os

# Environment variables (replace with actual values in production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
DATA_PATH = os.getenv("DATA_PATH", "telecom_dataset.csv")
MODELS_DIR = os.getenv("MODELS_DIR", "models")

# Data file paths (in order of preference)
DATA_FILE_PATHS = [
    "telecom_dataset.csv",
    "telecom_dataset_generated.csv", 
    "telecom_dataset_preprocessed.csv",
    os.path.join("data", "telecom_dataset.csv")
]

# Model file paths
MODEL_FILES = {
    "churn_model": "enhanced_churn_model.pkl",
    "revenue_model": "revenue_forecasting.pkl",
    "scaler": "scaler.pkl",
    "imputer": "imputer.pkl",
    "train_columns": "train_columns.pkl"
}

# User preferences (to be saved/loaded from a database in a real application)
DEFAULT_THEME = "light"
DEFAULT_LANGUAGE = "en"

# Chatbot configuration
CHATBOT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_CHATBOT_HISTORY = 10

# Security settings
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}
MAX_UPLOAD_SIZE_MB = 50

# Application settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///telecom_analytics.db")

# Feature columns for models
CHURN_FEATURES = [
    "Age", "MonthlyIncome", "MonthlySpending", "NetworkQuality",
    "DataUsageGB", "CallUsageMin", "SMSUsage", "CustomerCareCalls",
    "TenureMonths", "ServiceRating"
]

REVENUE_FEATURES = [
    "MonthlyIncome", "MonthlySpending", "DataUsageGB", "CallUsageMin",
    "SMSUsage", "TenureMonths", "ServiceRating"
]
