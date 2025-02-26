import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
import joblib
import optuna 
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class OptimizedRevenueForecaster:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.ensemble = self.create_ensemble()
        self.feature_importance: Dict[str, float] = {}
        self.arima_model = None

    def create_ensemble(self) -> VotingRegressor:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        xgb = XGBRegressor(n_estimators=100, random_state=42)
        lgbm = LGBMRegressor(n_estimators=100, random_state=42)
        lr = LinearRegression()

        return VotingRegressor(
            estimators=[('rf', rf), ('gb', gb), ('xgb', xgb), ('lgbm', lgbm), ('lr', lr)]
        )

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            model = RandomForestRegressor(**params, random_state=42)
            score = -np.mean(cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))
            return score

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Label encode binary categorical columns
        binary_cols = ['Gender', 'AdditionalServices', 'InternationalPlan', 'SubscriptionType']
        le = LabelEncoder()
        for col in binary_cols:
            df[col] = le.fit_transform(df[col])

        # One-hot encode multi-class categorical columns
        multi_class_cols = ['Region', 'PaymentMethod', 'ContractType']
        df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)
        return df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = y.fillna(y.mean())  
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        best_params = self.optimize_hyperparameters(X_scaled, y)
        logging.info(f"Optimized Hyperparameters: {best_params}")
        rf = RandomForestRegressor(**best_params, random_state=42)

        self.ensemble.estimators_ = [
            (name, rf if name == 'rf' else model) for name, model in self.ensemble.estimators
        ]
        self.ensemble.fit(X_scaled, y)
        
        # ARIMA model fitting
        try:
            self.arima_model = ARIMA(y, order=(1, 1, 1)).fit()
        except Exception as e:
            logging.warning(f"Failed to fit ARIMA model: {str(e)}")
            self.arima_model = None

        self.feature_importance = self.calculate_feature_importance(X)

        return self

    def calculate_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        feature_importance = {col: 0 for col in X.columns}
        for name, model in self.ensemble.estimators:
            if hasattr(model, 'feature_importances_'):
                for i, importance in enumerate(model.feature_importances_):
                    feature_importance[X.columns[i]] += importance

        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            return {k: v / total_importance for k, v in feature_importance.items()}
        else:
            logging.warning("Total feature importance is zero. Returning equal importance for all features.")
            return {k: 1.0 / len(feature_importance) for k in feature_importance.keys()}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        ensemble_preds = self.ensemble.predict(X_scaled)
        
        if self.arima_model:
            try:
                arima_preds = self.arima_model.forecast(steps=len(X))
                return 0.7 * ensemble_preds + 0.3 * arima_preds
            except Exception as e:
                logging.warning(f"Failed to generate ARIMA predictions: {str(e)}")
        
        return ensemble_preds

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        y = y.fillna(y.mean()) 
        y_pred = self.predict(X)
        return {
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mape': mean_absolute_percentage_error(y, y_pred)
        }

    def plot_feature_importance(self):
        plt.figure(figsize=(10, 6))
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        plt.bar(features, importances)
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str):
        joblib.dump({
            'ensemble': self.ensemble,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'arima_model': self.arima_model,
            'feature_importance': self.feature_importance
        }, filepath)

    @classmethod
    def load_model(cls, filepath: str):
        data = joblib.load(filepath)
        model = cls()
        model.ensemble = data['ensemble']
        model.scaler = data['scaler']
        model.imputer = data['imputer']
        model.arima_model = data['arima_model']
        model.feature_importance = data['feature_importance']
        return model

# Example usage
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('telecom_dataset.csv') 
    
    # Preprocess data
    model = OptimizedRevenueForecaster()
    df = model.preprocess_data(df)

    # Define features and target
    features = ['Age', 'MonthlyIncome', 'NetworkQuality', 'TenureMonths', 'ServiceRating', 
                'MonthlySpending', 'DataUsageGB', 'CallUsageMin', 'SMSUsage', 
                'CustomerCareCalls', 'ChurnRiskScore']
    X = df[features]
    y = df['MonthlySpending']  # MonthlySpending is the target for revenue forecasting

    # Split the data according to the 70-20-10 rule
    train_mask = df['DataSplit'] == 'Train'
    val_mask = df['DataSplit'] == 'Validation'
    test_mask = df['DataSplit'] == 'Test'

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_evaluation = model.evaluate(X_val, y_val)
    logging.info(f"Validation Metrics: {val_evaluation}")

    # Evaluate on test set
    test_evaluation = model.evaluate(X_test, y_test)
    logging.info(f"Test Metrics: {test_evaluation}")

    # Plot feature importance
    model.plot_feature_importance()

    # Save the model
    model.save_model('models/revenue_forecasting.pkl')

    # Make predictions for the next 30 days (each row represents a day)
    future_data = X_test.iloc[:30].copy()
    future_predictions = model.predict(future_data)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.iloc[:30].values, label='Actual')
    plt.plot(future_predictions, label='Predicted')
    plt.title('Revenue Forecast for Next 30 Days')
    plt.xlabel('Days')
    plt.ylabel('Revenue')
    plt.legend()
    plt.show()

    print("Revenue forecasting model training, evaluation, and prediction completed.")

