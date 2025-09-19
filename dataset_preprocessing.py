import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset with error handling
try:
    if os.path.exists("telecom_dataset.csv"):
        df = pd.read_csv("telecom_dataset.csv")
    elif os.path.exists("telecom_dataset_generated.csv"):
        df = pd.read_csv("telecom_dataset_generated.csv")
    else:
        print("No dataset found. Please run dataset_generator.py first.")
        exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Display initial info
print("Dataset Info:")
print(df.info())
print("\nUnique values in categorical columns:")

# Identify categorical and numerical columns
categorical_columns = ['Gender', 'Region', 'SubscriptionType', 'PaymentMethod', 
                       'ContractType', 'AdditionalServices', 'InternationalPlan', 'DataSplit']
numerical_columns = [col for col in df.columns if col not in categorical_columns + ['CustomerID']]

# Display unique values in categorical columns
for col in categorical_columns:
    print(f"\nUnique values in column {col}:")
    print(df[col].unique())

# Handle missing values
imputer = SimpleImputer(strategy="mean")
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False, drop="first")  # `sparse_output` instead of `sparse`
encoded_features = pd.DataFrame(
    encoder.fit_transform(df[categorical_columns]),
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Combine encoded features with the original dataframe
df_processed = pd.concat([df[numerical_columns], encoded_features], axis=1)

# Scale numerical features
scaler = StandardScaler()
df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])

# Save preprocessed data
df_processed.to_csv("telecom_dataset_preprocessed.csv", index=False)

# Display summary
print("\nPreprocessing completed. Saved as 'telecom_dataset_preprocessed.csv'.")
print("Processed dataset info:")
print(df_processed.info())
