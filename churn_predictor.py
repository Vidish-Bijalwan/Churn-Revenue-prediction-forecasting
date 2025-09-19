import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import os

class EnhancedChurnPredictor:
    def __init__(self, data_path: str):
        """
        Initialize the churn predictor with dataset path.
        Args:
            data_path (str): Path to the preprocessed telecom dataset.
        """
        self.data_path = data_path
        self.model = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.train_columns = None  # To store the columns after preprocessing

    def load_data(self):
        """Load and preprocess the dataset."""
        df = pd.read_csv(self.data_path)

        # Split dataset into features and target
        X = df.drop("Churned", axis=1)
        y = df["Churned"]

        # Recode the target variable to binary if needed
        y = y.apply(lambda x: 1 if x > 0 else 0)

        # Data Preprocessing
        numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = X.select_dtypes(include=["object"]).columns

        # Impute missing values
        X[numerical_cols] = self.imputer.fit_transform(X[numerical_cols])
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy="most_frequent")
            X[categorical_cols] = imputer.fit_transform(X[categorical_cols])

        # One-hot encode categorical features
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, drop_first=True)

        # Save the columns after preprocessing for alignment during prediction
        self.train_columns = X.columns

        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train_model(self):
        """Train the churn prediction model."""
        X, y = self.load_data()

        # Train-Test Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize XGBoost Classifier
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "scale_pos_weight": [1, 10],
        }
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Best model after grid search
        self.model = grid_search.best_estimator_

        # Evaluate the model
        y_pred = self.model.predict(X_val)
        print("Classification Report:\n", classification_report(y_val, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_val, y_pred))

        # Save the trained model, imputer, scaler, and columns
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(models_dir, "enhanced_churn_model.pkl"))
        joblib.dump(self.imputer, os.path.join(models_dir, "imputer.pkl"))
        joblib.dump(self.scaler, os.path.join(models_dir, "scaler.pkl"))
        joblib.dump(self.train_columns, os.path.join(models_dir, "train_columns.pkl"))

    def preprocess_input_data(self, input_data):
        """
        Preprocess input data to match training features.
        Args:
            input_data (pd.DataFrame): New input data for prediction.
        Returns:
            np.array: Preprocessed and scaled input data.
        """
        # Impute missing values
        input_data = input_data.fillna(0)  # Replace missing values with 0

        # One-hot encode categorical variables
        input_data_encoded = pd.get_dummies(input_data)

        # Load the train_columns to align features
        train_columns_path = os.path.join("models", "train_columns.pkl")
        train_columns = joblib.load(train_columns_path)

        # Add missing columns with default value 0
        missing_cols = set(train_columns) - set(input_data_encoded.columns)
        for col in missing_cols:
            input_data_encoded[col] = 0  # Add missing columns with default value 0

        # Remove extra columns in case there are any
        input_data_encoded = input_data_encoded.reindex(columns=train_columns, fill_value=0)

        # Debugging information - Comment this out to suppress debug messages
        # print("Missing columns added:", missing_cols)
        # print("Final input shape after alignment:", input_data_encoded.shape)

        # Scale numerical features
        scaler_path = os.path.join("models", "scaler.pkl")
        scaler = joblib.load(scaler_path)
        input_data_scaled = scaler.transform(input_data_encoded)

        return input_data_scaled

    def predict_churn(self, input_data):
        """
        Predict churn risk for new input data.
        Args:
            input_data (dict): Input data as a dictionary.
        Returns:
            np.array: Predicted churn probabilities.
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data
        input_scaled = self.preprocess_input_data(input_df)

        # Perform prediction
        prediction = self.model.predict_proba(input_scaled)[:, 1]  # Probability of churn
        return prediction

    def plot_feature_importance(self):
        """Plot the feature importance."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        xgb.plot_importance(self.model)
        plt.show()


# To train the model (only run if this file is executed directly)
if __name__ == "__main__":
    churn_predictor = EnhancedChurnPredictor(data_path="telecom_dataset_preprocessed.csv")
    churn_predictor.train_model()
