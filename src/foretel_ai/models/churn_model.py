"""
Professional churn prediction model for ForeTel.AI.

This module provides a comprehensive churn prediction system using XGBoost
with advanced feature engineering, hyperparameter optimization, and model explainability.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import shap

from ..config.settings import config, ModelType
from ..utils.logger import get_logger, log_model_metrics, log_performance
from ..data.processor import TelecomDataProcessor

logger = get_logger(__name__)

@dataclass
class ChurnPredictionResult:
    """Result of churn prediction."""
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    confidence: float
    feature_importance: Dict[str, float]

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str

class ChurnPredictor:
    """
    Advanced churn prediction model using XGBoost.
    
    Features:
    - Automated hyperparameter tuning
    - Feature importance analysis
    - Model explainability with SHAP
    - Cross-validation and robust evaluation
    - Production-ready prediction pipeline
    """
    
    def __init__(self):
        """Initialize the churn predictor."""
        self.model: Optional[xgb.XGBClassifier] = None
        self.data_processor = TelecomDataProcessor()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.model_metrics: Optional[ModelMetrics] = None
        self.shap_explainer: Optional[shap.Explainer] = None
        
        logger.info("ChurnPredictor initialized")
    
    def train(
        self, 
        df: pd.DataFrame,
        target_column: str = "Churned",
        optimize_hyperparameters: bool = True,
        save_model: bool = True
    ) -> ModelMetrics:
        """
        Train the churn prediction model.
        
        Args:
            df: Training dataset
            target_column: Name of target column
            optimize_hyperparameters: Whether to optimize hyperparameters
            save_model: Whether to save the trained model
            
        Returns:
            ModelMetrics with training performance
        """
        logger.info("Starting churn model training")
        
        # Preprocess data
        X_train, X_test, y_train, y_test = self.data_processor.preprocess_data(
            df, target_column, config.models.test_size, config.models.random_state
        )
        
        self.feature_names = X_train.columns.tolist()
        
        # Initialize model
        if optimize_hyperparameters:
            self.model = self._optimize_hyperparameters(X_train, y_train)
        else:
            self.model = xgb.XGBClassifier(
                random_state=config.models.random_state,
                eval_metric='logloss'
            )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        self.model_metrics = self._evaluate_model(X_test, y_test)
        
        # Setup SHAP explainer
        self._setup_shap_explainer(X_train.sample(min(1000, len(X_train))))
        
        self.is_trained = True
        
        # Log metrics
        log_model_metrics("ChurnPredictor", {
            "accuracy": self.model_metrics.accuracy,
            "precision": self.model_metrics.precision,
            "recall": self.model_metrics.recall,
            "f1_score": self.model_metrics.f1_score,
            "roc_auc": self.model_metrics.roc_auc
        })
        
        # Save model
        if save_model:
            self.save_model()
        
        logger.info("Churn model training completed successfully")
        return self.model_metrics
    
    def predict(
        self, 
        data: Union[pd.DataFrame, Dict[str, Any]],
        explain: bool = True
    ) -> Union[ChurnPredictionResult, List[ChurnPredictionResult]]:
        """
        Make churn predictions for new data.
        
        Args:
            data: Input data (DataFrame or dict)
            explain: Whether to include feature explanations
            
        Returns:
            ChurnPredictionResult(s) with predictions and explanations
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert single dict to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
            single_prediction = True
        else:
            df = data.copy()
            single_prediction = False
        
        # Preprocess data
        X = self.data_processor.transform_new_data(df)
        
        # Ensure feature alignment
        X = self._align_features(X)
        
        # Make predictions
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)
        
        # Generate results
        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            # Calculate risk level and confidence
            risk_level = self._calculate_risk_level(prob)
            confidence = max(prob, 1 - prob)
            
            # Get feature importance for this prediction
            feature_importance = {}
            if explain and self.shap_explainer:
                try:
                    shap_values = self.shap_explainer(X.iloc[[i]])
                    feature_importance = dict(zip(
                        self.feature_names,
                        shap_values.values[0]
                    ))
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
                    feature_importance = self._get_global_feature_importance()
            
            result = ChurnPredictionResult(
                customer_id=df.iloc[i].get('CustomerID', f'Customer_{i}'),
                churn_probability=float(prob),
                churn_prediction=bool(pred),
                risk_level=risk_level,
                confidence=float(confidence),
                feature_importance=feature_importance
            )
            results.append(result)
        
        return results[0] if single_prediction else results
    
    def get_feature_importance(self, plot: bool = False) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Args:
            plot: Whether to create a plot
            
        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_dict = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        if plot:
            self._plot_feature_importance(importance_dict)
        
        return importance_dict
    
    def save_model(self, file_path: Optional[str] = None) -> str:
        """
        Save the trained model and preprocessors.
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if file_path is None:
            file_path = config.get_model_path(ModelType.CHURN_PREDICTION)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'data_processor': self.data_processor,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics,
            'config_version': config.version
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Churn model saved to {file_path}")
        
        return str(file_path)
    
    def load_model(self, file_path: Optional[str] = None) -> bool:
        """
        Load a trained model.
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if file_path is None:
            file_path = config.get_model_path(ModelType.CHURN_PREDICTION)
        
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.data_processor = model_data['data_processor']
            self.feature_names = model_data['feature_names']
            self.model_metrics = model_data.get('model_metrics')
            
            # Setup SHAP explainer if model is loaded
            if self.model is not None:
                self._setup_shap_explainer()
            
            self.is_trained = True
            logger.info(f"Churn model loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load churn model: {e}")
            return False
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Optimize model hyperparameters using GridSearchCV."""
        logger.info("Optimizing hyperparameters")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=config.models.random_state,
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=config.models.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate model performance."""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1_score=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_pred_proba),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            classification_report=classification_report(y_test, y_pred)
        )
        
        return metrics
    
    def _setup_shap_explainer(self, background_data: Optional[pd.DataFrame] = None) -> None:
        """Setup SHAP explainer for model interpretability."""
        try:
            if background_data is not None:
                self.shap_explainer = shap.TreeExplainer(self.model, background_data)
            else:
                self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"Failed to setup SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align features with training data."""
        # Add missing columns with zeros
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Remove extra columns and reorder
        X = X[self.feature_names]
        
        return X
    
    def _calculate_risk_level(self, probability: float) -> str:
        """Calculate risk level based on churn probability."""
        if probability >= 0.8:
            return "Very High"
        elif probability >= 0.6:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        elif probability >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _get_global_feature_importance(self) -> Dict[str, float]:
        """Get global feature importance as fallback."""
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def _plot_feature_importance(self, importance_dict: Dict[str, float]) -> None:
        """Plot feature importance."""
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features[:15])  # Top 15 features
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances - Churn Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

__all__ = ['ChurnPredictor', 'ChurnPredictionResult', 'ModelMetrics']
