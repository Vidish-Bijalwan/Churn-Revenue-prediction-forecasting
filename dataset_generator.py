import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()

def generate_telecom_dataset(num_customers=10000):
    # Generate customer IDs
    customer_ids = [fake.unique.uuid4() for _ in range(num_customers)]
    
    # Generate basic customer information
    genders = np.random.choice(['Male', 'Female', 'Other'], num_customers, p=[0.49, 0.49, 0.02])
    ages = np.random.normal(40, 15, num_customers).astype(int)
    ages = np.clip(ages, 18, 90)  # Clip ages between 18 and 90
    
    # Generate location data
    regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_customers)
    
    # Generate subscription data
    subscription_types = np.random.choice(['Basic', 'Standard', 'Premium'], num_customers, p=[0.3, 0.5, 0.2])
    contract_types = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], num_customers, p=[0.6, 0.3, 0.1])
    tenure_months = np.random.gamma(shape=2, scale=15, size=num_customers).astype(int)
    tenure_months = np.clip(tenure_months, 1, 120)  # Clip tenure between 1 and 120 months
    
    # Generate usage data
    monthly_charges = np.zeros(num_customers)
    for i, sub_type in enumerate(subscription_types):
        if sub_type == 'Basic':
            monthly_charges[i] = np.random.uniform(20, 50)
        elif sub_type == 'Standard':
            monthly_charges[i] = np.random.uniform(50, 80)
        else:  # Premium
            monthly_charges[i] = np.random.uniform(80, 120)
    
    data_usage_gb = np.random.gamma(shape=2, scale=5, size=num_customers) * 10  # Scale up for more realistic usage
    minutes_used = np.random.gamma(shape=2, scale=100, size=num_customers)
    texts_sent = np.random.gamma(shape=2, scale=50, size=num_customers)
    
    # Generate additional services and features
    international_plan = np.random.choice([0, 1], num_customers, p=[0.8, 0.2])
    voice_mail_plan = np.random.choice([0, 1], num_customers, p=[0.7, 0.3])
    customer_service_calls = np.random.poisson(lam=1.5, size=num_customers)
    
    # Generate payment information
    payment_methods = np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check'], num_customers)
    
    # Calculate total charges based on tenure and monthly charges
    total_charges = monthly_charges * tenure_months
    
    # Generate churn data (assuming churn is influenced by various factors)
    churn_prob = 0.05 + 0.1 * (customer_service_calls / 10) + 0.1 * (np.random.rand(num_customers) < 0.2)
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.rand(num_customers) < churn_prob
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'Region': regions,
        'SubscriptionType': subscription_types,
        'ContractType': contract_types,
        'TenureMonths': tenure_months,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'DataUsageGB': data_usage_gb,
        'MinutesUsed': minutes_used,
        'TextsSent': texts_sent,
        'InternationalPlan': international_plan,
        'VoiceMailPlan': voice_mail_plan,
        'CustomerServiceCalls': customer_service_calls,
        'PaymentMethod': payment_methods,
        'Churn': churn
    })
    
    return df

# Generate the dataset
telecom_df = generate_telecom_dataset(num_customers=10000)

# Display dataset info
print("Dataset Info:")
print(telecom_df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(telecom_df.describe())

# Display unique values in categorical columns
categorical_columns = ['Gender', 'Region', 'SubscriptionType', 'ContractType', 'PaymentMethod']
for col in categorical_columns:
    print(f"\nUnique values in column {col}:")
    print(telecom_df[col].value_counts(normalize=True))

# Save the generated dataset
telecom_df.to_csv("telecom_dataset_generated.csv", index=False)
print("\nDataset generation completed. Saved as 'telecom_dataset_generated.csv'.")

