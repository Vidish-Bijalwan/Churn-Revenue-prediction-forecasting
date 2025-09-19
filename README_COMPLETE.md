# ForeTel.AI: Advanced Telecom Analytics Platform

ForeTel.AI is a comprehensive Python-based analytics platform designed for telecom companies to:
1. **Predict Customer Churn** using advanced machine learning models
2. **Forecast Revenue** with ensemble modeling techniques  
3. **Analyze Customer Behavior** through interactive dashboards
4. **Generate Actionable Insights** for business decision-making

---

## 🚀 Features

### **Core Analytics**
- **Churn Prediction**: XGBoost-based model with hyperparameter optimization
- **Revenue Forecasting**: Ensemble model combining Random Forest, Gradient Boosting, XGBoost, LightGBM, and Linear Regression
- **Interactive Dashboard**: Real-time analytics with Plotly visualizations
- **AI Assistant**: Rule-based chatbot for data queries

### **Data Management**
- **Synthetic Dataset Generation**: Realistic telecom customer data
- **Automated Data Preprocessing**: Missing value handling, encoding, scaling
- **Multiple Data Sources**: Support for CSV, Excel, and database connections
- **Data Validation**: Comprehensive error checking and validation

### **User Interface**
- **Streamlit Web App**: Professional, responsive interface
- **6 Main Sections**: Home, Churn Prediction, Revenue Forecasting, Analytics Dashboard, Chat Assistant, Document Insights
- **Custom Styling**: Modern CSS with mobile optimization
- **Real-time Updates**: Live data refresh and model predictions

### **Technical Excellence**
- **Error Handling**: Comprehensive exception management
- **Logging**: Detailed application and model logging
- **Performance Optimization**: Caching and efficient data processing
- **Cross-platform**: Works on Windows, macOS, and Linux

---

## 🛠 Installation

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### **Quick Start**
```bash
# Clone the repository
git clone https://github.com/your-username/ForeTel-AI.git
cd ForeTel-AI

# Install dependencies
pip install -r Requirements.txt

# Run the application
python run_app.py
```

### **Alternative Start Methods**
```bash
# Direct Streamlit run
streamlit run app.py

# Or with custom port
streamlit run app.py --server.port 8501
```

### **Development Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r Requirements.txt

# Generate sample data (if needed)
python dataset_generator.py

# Train models (optional)
python churn_predictor.py
python revenue_forecasting_model.py
```

---

## 📁 Project Structure

```
ForeTel-AI/
├── app.py                      # Main Streamlit application
├── run_app.py                  # Startup script with checks
├── config.py                   # Configuration settings
├── utils.py                    # Utility functions
├── Requirements.txt            # Python dependencies
├── 
├── churn_predictor.py          # Churn prediction model
├── revenue_forecasting_model.py # Revenue forecasting model
├── dataset_generator.py        # Synthetic data generation
├── dataset_preprocessing.py    # Data preprocessing pipeline
├── 
├── models/                     # Trained ML models
│   ├── enhanced_churn_model.pkl
│   ├── revenue_forecasting.pkl
│   ├── scaler.pkl
│   ├── imputer.pkl
│   └── train_columns.pkl
├── 
├── data/                       # Data files (auto-created)
├── logs/                       # Application logs (auto-created)
├── 
├── .env                        # Environment variables
├── model_metadata.json         # Model configuration
└── README.md                   # This file
```

---

## 🚀 Usage

### **1. Starting the Application**
```bash
python run_app.py
```
This will:
- Check system requirements
- Verify dependencies
- Create necessary directories
- Generate sample data if needed
- Start the Streamlit web interface

### **2. Accessing the Web Interface**
- Open your browser to `http://localhost:8501`
- Navigate through the 6 main sections using the sidebar

### **3. Main Features**

#### **Home Dashboard**
- Overview of key business metrics
- Customer segmentation analysis
- Revenue trends and insights
- AI-powered alerts and recommendations

#### **Churn Prediction**
- Input customer data through interactive forms
- Get real-time churn probability predictions
- View risk categorization (Low/Medium/High)
- Receive actionable recommendations

#### **Revenue Forecasting**
- Generate revenue forecasts for different time periods
- Interactive charts and trend analysis
- Model accuracy metrics and confidence intervals
- Export forecasts for business planning

#### **Analytics Dashboard**
- Comprehensive data exploration
- Customer behavior analysis
- Service usage patterns
- Performance metrics visualization

#### **Chat Assistant**
- Ask questions about your data
- Get instant insights and statistics
- Natural language query interface
- Historical conversation tracking

#### **Document Insights**
- Upload CSV files for analysis
- Automated data profiling
- Custom analytics and visualizations
- Export analysis reports

---

## 🔧 Configuration

### **Environment Variables**
Create a `.env` file in the project root:
```env
# Database Configuration
MONGO_URI=your_mongodb_connection_string

# Application Settings
DEBUG_MODE=False
LOG_LEVEL=INFO

# Model Paths
MODELS_DIR=models
DATA_PATH=telecom_dataset.csv

# API Keys (if using external services)
OPENAI_API_KEY=your_openai_key
```

### **Model Configuration**
Edit `model_metadata.json` to customize:
- Feature selection for models
- Default parameter values
- Model performance thresholds

---

## 📊 Data Schema

### **Required Columns**
The application expects the following columns in your dataset:

| Column | Type | Description |
|--------|------|-------------|
| CustomerID | String | Unique customer identifier |
| Age | Integer | Customer age |
| Gender | String | Customer gender |
| Region | String | Geographic region |
| TenureMonths | Integer | Months as customer |
| MonthlySpending | Float | Monthly spending amount |
| SubscriptionType | String | Basic/Standard/Premium |
| ContractType | String | Contract duration |
| NetworkQuality | Integer | Network quality rating (1-5) |
| DataUsageGB | Float | Monthly data usage |
| CallUsageMin | Float | Monthly call minutes |
| SMSUsage | Integer | Monthly SMS count |
| CustomerCareCalls | Integer | Support calls count |
| ServiceRating | Integer | Service rating (1-5) |
| Churned | Integer | Churn status (0/1) |

### **Optional Columns**
- PaymentMethod
- InternationalPlan
- VoiceMailPlan
- ChurnRiskScore
- TotalRevenue
- ARPU (Average Revenue Per User)

---

## 🤖 Model Details

### **Churn Prediction Model**
- **Algorithm**: XGBoost Classifier
- **Optimization**: GridSearchCV hyperparameter tuning
- **Features**: Demographics, usage patterns, service quality
- **Output**: Churn probability with confidence intervals
- **Performance**: ROC-AUC score, precision, recall, F1-score

### **Revenue Forecasting Model**
- **Algorithm**: Ensemble of 5 models (RF, GB, XGB, LGBM, LR)
- **Optimization**: Optuna hyperparameter optimization
- **Features**: Financial metrics, usage patterns, tenure
- **Output**: Revenue predictions with trend analysis
- **Performance**: R², RMSE, MAE, MAPE

---

## 🔍 Troubleshooting

### **Common Issues**

#### **1. Import Errors**
```bash
# Install missing dependencies
pip install -r Requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

#### **2. Data Loading Issues**
- Ensure CSV files are in the correct format
- Check file permissions
- Verify column names match expected schema
- Use the sample data generator if needed

#### **3. Model Loading Errors**
- Check if model files exist in `models/` directory
- Retrain models if files are corrupted
- Verify model file permissions

#### **4. Streamlit Issues**
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with specific port
streamlit run app.py --server.port 8502
```

#### **5. Memory Issues**
- Reduce dataset size for testing
- Close other applications
- Use data sampling for large datasets

### **Getting Help**
1. Check the logs in `logs/` directory
2. Enable debug mode in `.env` file
3. Run with verbose logging: `python run_app.py --debug`

---

## 🔄 Development

### **Adding New Features**
1. Create feature branch: `git checkout -b feature/new-feature`
2. Add your code following the existing structure
3. Update tests and documentation
4. Submit pull request

### **Model Training**
```bash
# Train churn prediction model
python churn_predictor.py

# Train revenue forecasting model
python revenue_forecasting_model.py

# Generate new dataset
python dataset_generator.py
```

### **Testing**
```bash
# Run basic functionality test
python -c "import app; print('App imports successfully')"

# Test model loading
python -c "from app import load_models; print('Models loaded:', len(load_models()))"
```

---

## 📈 Performance Optimization

### **For Large Datasets**
- Use data sampling for initial exploration
- Implement pagination for data display
- Use chunked data processing
- Enable Streamlit caching

### **For Production Deployment**
- Use production-grade database (PostgreSQL, MongoDB)
- Implement load balancing
- Add monitoring and alerting
- Use containerization (Docker)

---

## 🚀 Deployment

### **Local Development**
```bash
streamlit run app.py
```

### **Production Deployment**
1. **Docker Deployment**
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r Requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

2. **Cloud Deployment**
- Deploy to Streamlit Cloud
- Use Heroku with Procfile
- Deploy to AWS/GCP/Azure

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

## 🎯 Roadmap

### **Upcoming Features**
- [ ] Real-time data streaming
- [ ] Advanced ML models (Deep Learning)
- [ ] Multi-tenant support
- [ ] API endpoints for integration
- [ ] Mobile-responsive design improvements
- [ ] Advanced visualization options
- [ ] Automated model retraining
- [ ] Export capabilities (PDF reports)

### **Version History**
- **v2.0.0** - Complete refactor with enhanced error handling
- **v1.0.0** - Initial release with basic functionality

---

*ForeTel.AI - Empowering Telecom Analytics with AI* 🚀
