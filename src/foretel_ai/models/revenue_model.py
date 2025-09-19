"""
Professional revenue forecasting model for ForeTel.AI.

This module provides comprehensive revenue forecasting using ensemble methods,
time series analysis, and advanced optimization techniques.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import optuna
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import config, ModelType
from ..utils.logger import get_logger, log_model_metrics, log_performance
from ..data.processor import TelecomDataProcessor

logger = get_logger(__name__)

@dataclass
class RevenueForecast:
    """Revenue forecast result."""
    customer_id: str
    predicted_revenue: float
    confidence_interval: Tuple[float, float]
    forecast_period: str
    model_confidence: float
    contributing_factors: Dict[str, float]

@dataclass
class EnsembleMetrics:
    """Ensemble model performance metrics."""
    mse: float
    rmse: float
    mae: float
    r2_score: float
    mape: float
    individual_scores: Dict[str, float]

class RevenueForecaster:
    """
    Advanced revenue forecasting system using ensemble methods.
    
    Features:
    - Multiple ML algorithms (RF, GB, XGB, LGBM, LR)
    - Ensemble voting and stacking
    - Time series analysis with ARIMA
    - Hyperparameter optimization with Optuna
    - Confidence intervals and uncertainty quantification
    """
    
    def __init__(self):
        """Initialize the revenue forecaster."""
        self.models: Dict[str, Any] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.data_processor = TelecomDataProcessor()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.model_metrics: Optional[EnsembleMetrics] = None
        self.arima_model: Optional[ARIMA] = None
        
        logger.info("RevenueForecaster initialized")
    
    def train(
        self, 
        df: pd.DataFrame,
        target_column: str = "TotalRevenue",
        optimize_hyperparameters: bool = True,
        include_time_series: bool = True,
        save_model: bool = True
    ) -> EnsembleMetrics:
        """
        Train the revenue forecasting ensemble.
        
        Args:
            df: Training dataset
            target_column: Name of target column
            optimize_hyperparameters: Whether to optimize hyperparameters
            include_time_series: Whether to include ARIMA model
            save_model: Whether to save the trained model
            
        Returns:
            EnsembleMetrics with training performance
        """
        logger.info("Starting revenue forecasting model training")
        
        # Preprocess data
        X_train, X_test, y_train, y_test = self.data_processor.preprocess_data(
            df, target_column, config.models.test_size, config.models.random_state
        )
        
        self.feature_names = X_train.columns.tolist()
        
        # Train individual models
        if optimize_hyperparameters:
            self._train_optimized_models(X_train, y_train)
        else:
            self._train_default_models(X_train, y_train)
        
        # Train time series model if requested
        if include_time_series and 'TenureMonths' in df.columns:
            self._train_arima_model(df, target_column)
        
        # Calculate ensemble weights
        self._calculate_ensemble_weights(X_train, y_train)
        
        # Evaluate ensemble
        self.model_metrics = self._evaluate_ensemble(X_test, y_test)
        
        self.is_trained = True
        
        # Log metrics
        log_model_metrics("RevenueForecaster", {
            "rmse": self.model_metrics.rmse,
            "mae": self.model_metrics.mae,
            "r2_score": self.model_metrics.r2_score,
            "mape": self.model_metrics.mape
        })
        
        # Save model
        if save_model:
            self.save_model()
        
        logger.info("Revenue forecasting model training completed successfully")
        return self.model_metrics
    
    def predict(
        self, 
        data: Union[pd.DataFrame, Dict[str, Any]],
        include_confidence: bool = True,
        forecast_period: str = "monthly"
    ) -> Union[RevenueForecast, List[RevenueForecast]]:
        """
        Make revenue forecasts for new data.
        
        Args:
            data: Input data (DataFrame or dict)
            include_confidence: Whether to include confidence intervals
            forecast_period: Forecast period ("monthly", "quarterly", "yearly")
            
        Returns:
            RevenueForecast(s) with predictions and confidence intervals
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
        X = self._align_features(X)
        
        # Make ensemble predictions
        predictions = self._predict_ensemble(X)
        
        # Calculate confidence intervals
        confidence_intervals = []
        if include_confidence:
            confidence_intervals = self._calculate_confidence_intervals(X, predictions)
        
        # Generate results
        results = []
        for i, pred in enumerate(predictions):
            # Get feature contributions
            contributing_factors = self._get_feature_contributions(X.iloc[[i]])
            
            # Calculate model confidence
            model_confidence = self._calculate_model_confidence(X.iloc[[i]])
            
            # Confidence interval
            conf_interval = confidence_intervals[i] if confidence_intervals else (pred * 0.9, pred * 1.1)
            
            result = RevenueForecast(
                customer_id=df.iloc[i].get('CustomerID', f'Customer_{i}'),
                predicted_revenue=float(pred),
                confidence_interval=conf_interval,
                forecast_period=forecast_period,
                model_confidence=float(model_confidence),
                contributing_factors=contributing_factors
            )
            results.append(result)
        
        return results[0] if single_prediction else results
    
    def get_model_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from all models."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = dict(zip(
                    self.feature_names,
                    model.feature_importances_
                ))
            elif hasattr(model, 'coef_'):
                importance_dict[name] = dict(zip(
                    self.feature_names,
                    np.abs(model.coef_)
                ))
        
        return importance_dict
    
    def save_model(self, file_path: Optional[str] = None) -> str:
        """Save the trained ensemble model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if file_path is None:
            file_path = config.get_model_path(ModelType.REVENUE_FORECASTING)
        
        model_data = {
            'models': self.models,
            'ensemble_weights': self.ensemble_weights,
            'data_processor': self.data_processor,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics,
            'arima_model': self.arima_model,
            'config_version': config.version
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Revenue forecasting model saved to {file_path}")
        
        return str(file_path)
    
    def load_model(self, file_path: Optional[str] = None) -> bool:
        """Load a trained ensemble model."""
        if file_path is None:
            file_path = config.get_model_path(ModelType.REVENUE_FORECASTING)
        
        try:
            model_data = joblib.load(file_path)
            
            self.models = model_data['models']
            self.ensemble_weights = model_data['ensemble_weights']
            self.data_processor = model_data['data_processor']
            self.feature_names = model_data['feature_names']
            self.model_metrics = model_data.get('model_metrics')
            self.arima_model = model_data.get('arima_model')
            
            self.is_trained = True
            logger.info(f"Revenue forecasting model loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load revenue forecasting model: {e}")
            return False
    
    def _train_default_models(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train models with default parameters."""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=config.models.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=config.models.random_state
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=config.models.random_state,
                eval_metric='rmse'
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                random_state=config.models.random_state,
                verbose=-1
            ),
            'linear_regression': LinearRegression()
        }
        
        for name, model in self.models.items():
            logger.info(f"Training {name}")
            model.fit(X, y)
    
    def _train_optimized_models(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train models with optimized hyperparameters using Optuna."""
        logger.info("Optimizing hyperparameters with Optuna")
        
        self.models = {}
        
        # Random Forest optimization
        self.models['random_forest'] = self._optimize_random_forest(X, y)
        
        # XGBoost optimization
        self.models['xgboost'] = self._optimize_xgboost(X, y)
        
        # LightGBM optimization
        self.models['lightgbm'] = self._optimize_lightgbm(X, y)
        
        # Add non-optimized models
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            random_state=config.models.random_state
        )
        self.models['linear_regression'] = LinearRegression()
        
        # Train non-optimized models
        for name in ['gradient_boosting', 'linear_regression']:
            self.models[name].fit(X, y)
    
    def _optimize_random_forest(self, X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
        """Optimize Random Forest hyperparameters."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'random_state': config.models.random_state,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        best_model = RandomForestRegressor(**study.best_params)
        best_model.fit(X, y)
        
        return best_model
    
    def _optimize_xgboost(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
        """Optimize XGBoost hyperparameters."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': config.models.random_state,
                'eval_metric': 'rmse'
            }
            
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        best_model = xgb.XGBRegressor(**study.best_params)
        best_model.fit(X, y)
        
        return best_model
    
    def _optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
        """Optimize LightGBM hyperparameters."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': config.models.random_state,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        best_model = lgb.LGBMRegressor(**study.best_params)
        best_model.fit(X, y)
        
        return best_model
    
    def _train_arima_model(self, df: pd.DataFrame, target_column: str) -> None:
        """Train ARIMA model for time series forecasting."""
        try:
            # Create time series data
            if 'TenureMonths' in df.columns:
                ts_data = df.groupby('TenureMonths')[target_column].mean().sort_index()
                
                # Fit ARIMA model
                self.arima_model = ARIMA(ts_data, order=(1, 1, 1))
                self.arima_model = self.arima_model.fit()
                
                logger.info("ARIMA model trained successfully")
        except Exception as e:
            logger.warning(f"Failed to train ARIMA model: {e}")
            self.arima_model = None
    
    def _calculate_ensemble_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calculate optimal ensemble weights using cross-validation."""
        scores = {}
        
        for name, model in self.models.items():
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            scores[name] = -cv_scores.mean()  # Convert to positive MSE
        
        # Calculate weights inversely proportional to MSE
        total_inverse_mse = sum(1/mse for mse in scores.values())
        self.ensemble_weights = {
            name: (1/mse) / total_inverse_mse 
            for name, mse in scores.items()
        }
        
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
    
    def _predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = np.zeros(len(X))
        
        for name, model in self.models.items():
            weight = self.ensemble_weights.get(name, 1.0 / len(self.models))
            pred = model.predict(X)
            predictions += weight * pred
        
        return predictions
    
    def _evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series) -> EnsembleMetrics:
        """Evaluate ensemble performance."""
        # Ensemble predictions
        y_pred = self._predict_ensemble(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Individual model scores
        individual_scores = {}
        for name, model in self.models.items():
            pred = model.predict(X_test)
            individual_scores[name] = r2_score(y_test, pred)
        
        return EnsembleMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            mape=mape,
            individual_scores=individual_scores
        )
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align features with training data."""
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        return X[self.feature_names]
    
    def _calculate_confidence_intervals(self, X: pd.DataFrame, predictions: np.ndarray) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for predictions."""
        confidence_intervals = []
        
        for i, pred in enumerate(predictions):
            # Use ensemble variance as uncertainty measure
            individual_preds = []
            for model in self.models.values():
                individual_preds.append(model.predict(X.iloc[[i]])[0])
            
            std = np.std(individual_preds)
            lower = pred - 1.96 * std  # 95% confidence interval
            upper = pred + 1.96 * std
            
            confidence_intervals.append((float(lower), float(upper)))
        
        return confidence_intervals
    
    def _get_feature_contributions(self, X: pd.DataFrame) -> Dict[str, float]:
        """Get feature contributions for a prediction."""
        contributions = {}
        
        # Use Random Forest feature importance as proxy
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            if hasattr(rf_model, 'feature_importances_'):
                for i, feature in enumerate(self.feature_names):
                    contributions[feature] = float(rf_model.feature_importances_[i] * X.iloc[0, i])
        
        return contributions
    
    def _calculate_model_confidence(self, X: pd.DataFrame) -> float:
        """Calculate model confidence for a prediction."""
        # Use ensemble agreement as confidence measure
        individual_preds = []
        for model in self.models.values():
            individual_preds.append(model.predict(X)[0])
        
        # Calculate coefficient of variation (lower is better)
        mean_pred = np.mean(individual_preds)
        std_pred = np.std(individual_preds)
        
        if mean_pred != 0:
            cv = std_pred / abs(mean_pred)
            confidence = max(0, 1 - cv)  # Convert to confidence (0-1)
        else:
            confidence = 0.5
        
        return min(1.0, max(0.0, confidence))

__all__ = ['RevenueForecaster', 'RevenueForecast', 'EnsembleMetrics']
