"""
ForeTel.AI Enterprise Model Service
Advanced ML Pipeline with AutoML, Deep Learning, and Real-time Predictions
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
import h2o
from h2o.automl import H2OAutoML
import redis
import kafka
from kafka import KafkaProducer, KafkaConsumer
import asyncio
import logging
import os
from datetime import datetime
import json
import hashlib
from cryptography.fernet import Fernet
import shap
import lime
from lime import lime_tabular

# Configuration
REDIS_URL = os.getenv("REDIS_URL")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

# Initialize FastAPI
app = FastAPI(
    title="ForeTel.AI Model Service",
    description="Enterprise ML pipeline with AutoML and real-time predictions",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
redis_client = redis.from_url(REDIS_URL)
kafka_producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize H2O for AutoML
h2o.init()

# Encryption setup
cipher_suite = Fernet(ENCRYPTION_KEY.encode())

# Pydantic Models
class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_type: str  # "churn" or "revenue"
    explain: bool = False

class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, Any]]
    model_type: str
    explain: bool = False

class ModelTrainingRequest(BaseModel):
    dataset_path: str
    target_column: str
    model_type: str
    use_automl: bool = True
    hyperparameter_tuning: bool = True
    cross_validation_folds: int = 5

class ModelResponse(BaseModel):
    prediction: Any
    probability: Optional[float] = None
    confidence: Optional[float] = None
    explanation: Optional[Dict] = None
    model_version: str
    timestamp: datetime

class TrainingResponse(BaseModel):
    model_id: str
    model_version: str
    performance_metrics: Dict[str, float]
    training_time: float
    status: str

# Advanced ML Pipeline Classes
class AutoMLPipeline:
    def __init__(self):
        self.models = {}
        self.explainers = {}
    
    def train_automl_model(self, df: pd.DataFrame, target_column: str, model_type: str):
        """Train model using H2O AutoML"""
        h2o_df = h2o.H2OFrame(df)
        
        # Split data
        train, test = h2o_df.split_frame(ratios=[0.8], seed=42)
        
        # Configure AutoML
        aml = H2OAutoML(
            max_models=20,
            seed=42,
            max_runtime_secs=3600,  # 1 hour max
            sort_metric="AUC" if model_type == "churn" else "RMSE"
        )
        
        # Train
        aml.train(
            x=list(df.columns[df.columns != target_column]),
            y=target_column,
            training_frame=train
        )
        
        # Get best model
        best_model = aml.leader
        
        # Evaluate
        performance = best_model.model_performance(test)
        
        return best_model, performance

class DeepLearningPipeline:
    def __init__(self):
        self.models = {}
    
    def create_neural_network(self, input_shape: int, model_type: str):
        """Create deep learning model"""
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
        ])
        
        if model_type == "churn":
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
        else:  # revenue
            model.add(keras.layers.Dense(1, activation='linear'))
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae', 'mape']
            )
        
        return model
    
    def train_deep_model(self, X_train, y_train, X_val, y_val, model_type: str):
        """Train deep learning model"""
        model = self.create_neural_network(X_train.shape[1], model_type)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                f'models/deep_{model_type}_model.h5',
                save_best_only=True
            )
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        return model, history

class ExplainabilityEngine:
    def __init__(self):
        self.shap_explainers = {}
        self.lime_explainers = {}
    
    def create_shap_explainer(self, model, X_train, model_type: str):
        """Create SHAP explainer"""
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X_train.sample(100))
        
        self.shap_explainers[model_type] = explainer
        return explainer
    
    def create_lime_explainer(self, X_train, feature_names, model_type: str):
        """Create LIME explainer"""
        explainer = lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            mode='classification' if model_type == 'churn' else 'regression'
        )
        
        self.lime_explainers[model_type] = explainer
        return explainer
    
    def explain_prediction(self, model, instance, model_type: str, method: str = "shap"):
        """Generate explanation for prediction"""
        if method == "shap" and model_type in self.shap_explainers:
            shap_values = self.shap_explainers[model_type].shap_values(instance)
            return {
                "method": "shap",
                "feature_importance": shap_values.tolist(),
                "base_value": self.shap_explainers[model_type].expected_value
            }
        elif method == "lime" and model_type in self.lime_explainers:
            explanation = self.lime_explainers[model_type].explain_instance(
                instance.flatten(), model.predict
            )
            return {
                "method": "lime",
                "feature_importance": explanation.as_list()
            }
        else:
            return {"method": "none", "message": "No explainer available"}

# Initialize pipelines
automl_pipeline = AutoMLPipeline()
deep_learning_pipeline = DeepLearningPipeline()
explainability_engine = ExplainabilityEngine()

# Model cache
model_cache = {}
performance_cache = {}

# Utility functions
def encrypt_sensitive_data(data: str) -> str:
    """Encrypt sensitive data"""
    return cipher_suite.encrypt(data.encode()).decode()

def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive data"""
    return cipher_suite.decrypt(encrypted_data.encode()).decode()

def cache_model_prediction(key: str, prediction: Any, ttl: int = 3600):
    """Cache prediction in Redis"""
    redis_client.setex(key, ttl, json.dumps(prediction, default=str))

def get_cached_prediction(key: str):
    """Get cached prediction from Redis"""
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None

def send_to_kafka(topic: str, message: Dict):
    """Send message to Kafka"""
    kafka_producer.send(topic, message)

def generate_model_id() -> str:
    """Generate unique model ID"""
    return hashlib.sha256(f"{datetime.now()}{os.urandom(16)}".encode()).hexdigest()[:16]

# Routes
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "model",
        "timestamp": datetime.utcnow(),
        "redis_connected": redis_client.ping(),
        "mlflow_uri": MLFLOW_TRACKING_URI
    }

@app.post("/predict", response_model=ModelResponse)
async def predict(request: PredictionRequest):
    """Make single prediction"""
    # Create cache key
    cache_key = f"prediction:{request.model_type}:{hashlib.md5(str(request.features).encode()).hexdigest()}"
    
    # Check cache
    cached_result = get_cached_prediction(cache_key)
    if cached_result:
        return ModelResponse(**cached_result)
    
    try:
        # Load model
        if request.model_type not in model_cache:
            model_path = f"models/{request.model_type}_model.pkl"
            model_cache[request.model_type] = joblib.load(model_path)
        
        model = model_cache[request.model_type]
        
        # Prepare features
        feature_df = pd.DataFrame([request.features])
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            prediction = model.predict(feature_df)[0]
            probability = model.predict_proba(feature_df)[0][1]
        else:
            prediction = model.predict(feature_df)[0]
            probability = None
        
        # Generate explanation if requested
        explanation = None
        if request.explain:
            explanation = explainability_engine.explain_prediction(
                model, feature_df.values, request.model_type
            )
        
        # Create response
        response = ModelResponse(
            prediction=float(prediction),
            probability=float(probability) if probability is not None else None,
            confidence=0.95,  # Calculate actual confidence
            explanation=explanation,
            model_version="2.0.0",
            timestamp=datetime.utcnow()
        )
        
        # Cache result
        cache_model_prediction(cache_key, response.dict())
        
        # Send to Kafka for monitoring
        send_to_kafka("model_predictions", {
            "model_type": request.model_type,
            "prediction": float(prediction),
            "timestamp": datetime.utcnow().isoformat(),
            "features": request.features
        })
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions"""
    try:
        # Load model
        if request.model_type not in model_cache:
            model_path = f"models/{request.model_type}_model.pkl"
            model_cache[request.model_type] = joblib.load(model_path)
        
        model = model_cache[request.model_type]
        
        # Prepare features
        feature_df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model.predict(feature_df)
        probabilities = None
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_df)[:, 1]
        
        # Generate explanations if requested
        explanations = []
        if request.explain:
            for i, row in feature_df.iterrows():
                explanation = explainability_engine.explain_prediction(
                    model, row.values.reshape(1, -1), request.model_type
                )
                explanations.append(explanation)
        
        # Create response
        results = []
        for i, prediction in enumerate(predictions):
            result = {
                "prediction": float(prediction),
                "probability": float(probabilities[i]) if probabilities is not None else None,
                "explanation": explanations[i] if explanations else None,
                "index": i
            }
            results.append(result)
        
        # Send to Kafka for monitoring
        send_to_kafka("batch_predictions", {
            "model_type": request.model_type,
            "batch_size": len(predictions),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"results": results, "total_predictions": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train new model"""
    model_id = generate_model_id()
    
    # Start training in background
    background_tasks.add_task(
        _train_model_background,
        model_id,
        request.dataset_path,
        request.target_column,
        request.model_type,
        request.use_automl,
        request.hyperparameter_tuning
    )
    
    return TrainingResponse(
        model_id=model_id,
        model_version="2.0.0",
        performance_metrics={},
        training_time=0.0,
        status="training_started"
    )

async def _train_model_background(model_id: str, dataset_path: str, target_column: str,
                                model_type: str, use_automl: bool, hyperparameter_tuning: bool):
    """Background task for model training"""
    try:
        # Load data
        df = pd.read_csv(dataset_path)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_type}_training_{model_id}"):
            start_time = datetime.now()
            
            if use_automl:
                # Use AutoML
                model, performance = automl_pipeline.train_automl_model(df, target_column, model_type)
                
                # Log to MLflow
                mlflow.log_param("method", "automl")
                mlflow.log_param("model_type", model_type)
                
            else:
                # Use traditional ML or Deep Learning
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if model_type == "churn":
                    # Deep learning for churn
                    model, history = deep_learning_pipeline.train_deep_model(
                        X_train.values, y_train.values, X_test.values, y_test.values, model_type
                    )
                    
                    # Evaluate
                    predictions = model.predict(X_test.values)
                    accuracy = accuracy_score(y_test, (predictions > 0.5).astype(int))
                    
                    performance = {"accuracy": accuracy}
                    
                else:  # revenue
                    # Ensemble for revenue
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    predictions = model.predict(X_test)
                    mse = np.mean((y_test - predictions) ** 2)
                    
                    performance = {"mse": mse}
                
                # Log to MLflow
                mlflow.log_param("method", "deep_learning" if model_type == "churn" else "ensemble")
                mlflow.log_param("model_type", model_type)
                
                # Log model
                if hasattr(model, 'save'):  # TensorFlow model
                    mlflow.tensorflow.log_model(model, f"{model_type}_model")
                else:  # Scikit-learn model
                    mlflow.sklearn.log_model(model, f"{model_type}_model")
            
            # Log performance metrics
            for metric, value in performance.items():
                mlflow.log_metric(metric, value)
            
            # Save model
            model_path = f"models/{model_type}_model_{model_id}.pkl"
            joblib.dump(model, model_path)
            
            # Create explainer
            if not use_automl:  # AutoML models have built-in explanations
                explainability_engine.create_shap_explainer(model, X_train, model_type)
                explainability_engine.create_lime_explainer(X_train, X.columns.tolist(), model_type)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Update cache
            performance_cache[model_id] = {
                "model_id": model_id,
                "model_version": "2.0.0",
                "performance_metrics": performance,
                "training_time": training_time,
                "status": "completed"
            }
            
            # Send completion notification
            send_to_kafka("model_training_completed", {
                "model_id": model_id,
                "model_type": model_type,
                "performance": performance,
                "training_time": training_time,
                "timestamp": datetime.utcnow().isoformat()
            })
            
    except Exception as e:
        # Log error
        performance_cache[model_id] = {
            "model_id": model_id,
            "status": "failed",
            "error": str(e)
        }
        
        send_to_kafka("model_training_failed", {
            "model_id": model_id,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })

@app.get("/training/status/{model_id}")
async def get_training_status(model_id: str):
    """Get training status"""
    if model_id in performance_cache:
        return performance_cache[model_id]
    else:
        return {"status": "not_found", "message": "Model ID not found"}

@app.get("/models")
async def list_models():
    """List available models"""
    # Get from MLflow
    client = mlflow.tracking.MlflowClient()
    experiments = client.list_experiments()
    
    models = []
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)
        for run in runs:
            models.append({
                "run_id": run.info.run_id,
                "model_type": run.data.params.get("model_type", "unknown"),
                "method": run.data.params.get("method", "unknown"),
                "metrics": run.data.metrics,
                "start_time": run.info.start_time,
                "status": run.info.status
            })
    
    return {"models": models}

@app.post("/models/{model_id}/deploy")
async def deploy_model(model_id: str):
    """Deploy model to production"""
    try:
        # Load model from MLflow
        model_uri = f"runs:/{model_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Update production cache
        model_type = "churn"  # Get from metadata
        model_cache[f"{model_type}_production"] = model
        
        # Send deployment notification
        send_to_kafka("model_deployed", {
            "model_id": model_id,
            "model_type": model_type,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"status": "deployed", "model_id": model_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

@app.get("/performance/drift")
async def check_model_drift():
    """Check for model drift"""
    # This would implement statistical tests for data/concept drift
    # For now, return mock data
    return {
        "churn_model": {
            "data_drift": 0.05,
            "concept_drift": 0.02,
            "performance_degradation": 0.01,
            "recommendation": "retrain_recommended"
        },
        "revenue_model": {
            "data_drift": 0.03,
            "concept_drift": 0.01,
            "performance_degradation": 0.005,
            "recommendation": "monitoring_continue"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
