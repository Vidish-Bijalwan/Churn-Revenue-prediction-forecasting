import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
from streamlit_option_menu import option_menu
import os
from datetime import datetime, timedelta
import time
import base64
from typing import Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration and Setup
st.set_page_config(page_title="ForeTel.AI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #1e3a8a;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        color: white;
    }
    .metric-card h3 {
        margin-top: 0;
        color: #e0e7ff;
    }
    .metric-card-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .welcome-message {
        background-color: #1e40af;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 2rem;
        color: white;
    }
    .alert-high {
        color: #991b1b;
        background-color: #fee2e2;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .alert-positive {
        color: #065f46;
        background-color: #d1fae5;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .alert-warning {
        color: #92400e;
        background-color: #fef3c7;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .stButton > button {
        background-color: #2563eb;
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
    }
</style>
""", unsafe_allow_html=True)

# Caching and Loading Functions
@st.cache_data
def load_data():
    """
    Load and preprocess the telecom dataset.
    
    Returns:
        pd.DataFrame: Preprocessed telecom dataset
    """
    try:
        file_path = os.getenv('DATA_PATH', 'telecom_dataset.csv')
        df = pd.read_csv(file_path)
        df['TotalRevenue'] = df['MonthlySpending'] * df['TenureMonths']
        df['ARPU'] = df['TotalRevenue'] / df['TenureMonths']
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        st.error(f"Failed to load dataset. Please check if the file exists and is accessible.")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    """
    Load machine learning models for churn prediction and revenue forecasting.
    
    Returns:
        dict: Dictionary containing loaded models
    """
    try:
        models_dir = os.getenv('MODELS_DIR', 'models')
        churn_model = joblib.load(os.path.join(models_dir, 'enhanced_churn_model.pkl'))
        revenue_model = joblib.load(os.path.join(models_dir, 'revenue_forecasting.pkl'))
        return {
            'churn': churn_model,
            'revenue': revenue_model
        }
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        st.error("Failed to load models. Please check if model files exist and are accessible.")
        return {}

# Load data and models
df = load_data()
models = load_models()

# Utility Functions
def get_real_metrics(df):
    """
    Calculate real metrics from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        dict: Dictionary containing calculated metrics
    """
    try:
        total_customers = len(df)
        churn_rate = (df["Churned"].sum() / total_customers) * 100
        revenue = df["TotalRevenue"].sum()
        arpu = df["ARPU"].mean()

        return {
            "total_customers": total_customers,
            "churn_rate": round(churn_rate, 1),
            "revenue": int(revenue),
            "arpu": round(arpu, 2)
        }
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {str(e)}")
        return {}

def create_metric_card(title, value, delta, icon):
    """
    Create an HTML string for a metric card.
    
    Args:
        title (str): Title of the metric
        value (str): Value of the metric
        delta (str): Delta value of the metric
        icon (str): Icon for the metric
    
    Returns:
        str: HTML string for the metric card
    """
    delta_color = "green" if delta.startswith("+") else "red"
    return f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <div class="metric-card-value">{value}</div>
        <div style="color: {delta_color};">{delta}</div>
    </div>
    """

# Page Functions
def show_home_page(df):
    """
    Display the home page of the ForeTel.AI application.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    st.title("🏠 Telecom Analytics Platform")
    
    st.markdown("""
        <div class="welcome-message">
            <h2>Welcome to Advanced Telecom Analytics</h2>
            <p>Leverage AI-powered insights to optimize your telecom operations, predict customer behavior, 
            and maximize revenue potential.</p>
        </div>
    """, unsafe_allow_html=True)
    
    metrics = get_real_metrics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            "Total Customers",
            f"{metrics['total_customers']:,}",
            "+2.5%",
            "👥"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "Churn Rate",
            f"{metrics['churn_rate']}%",
            "-0.8%",
            "🔄"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "Revenue",
            f"{metrics['revenue']:,.0f}",
            "+12%",
            "💰"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card(
            "ARPU",
            f"{metrics['arpu']:.2f}",
            "+5%",
            "📈"
        ), unsafe_allow_html=True)
    
    # Business Insights Dashboard
    st.subheader("💡 Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_data = df['SubscriptionType'].value_counts().reset_index()
        segment_data.columns = ['Segment', 'Customers']
        segment_data['Growth'] = ['+12%', '+5%', '-2%'][:len(segment_data)]
        fig = px.bar(
            segment_data,
            x='Segment',
            y='Customers',
            text='Growth',
            title="Customer Segment Performance",
            color='Segment',
            color_discrete_map={'Premium': '#3b82f6', 'Standard': '#10b981', 'Basic': '#f59e0b'}
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        revenue_trend = df.groupby('TenureMonths')['TotalRevenue'].sum().reset_index()
        revenue_trend['TenureMonths'] = pd.to_datetime(revenue_trend['TenureMonths'], unit='M')
        fig = px.line(revenue_trend, x='TenureMonths', y='TotalRevenue', title="Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Insights Section
    st.subheader("🤖 AI-Powered Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <h3>Churn Risk Alert</h3>
                <p class="alert-high">High-value customers at risk</p>
                <p>23 Premium customers showing churn indicators</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("View Churn Risk Details"):
            show_churn_risk_details(df)
    
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <h3>Revenue Opportunity</h3>
                <p class="alert-positive">Upsell Potential</p>
                <p>156 customers eligible for premium upgrades</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Revenue Opportunities"):
            show_revenue_opportunities(df)
    
    with col3:
        st.markdown("""
            <div class='metric-card'>
                <h3>Network Performance</h3>
                <p class="alert-warning">Optimization Needed</p>
                <p>3 regions showing high latency patterns</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Investigate Network Performance"):
            show_network_performance(df)

def show_churn_risk_details(df):
    """
    Display details of high-risk customers for churn.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    st.subheader("Churn Risk Details")
    high_risk_customers = df[(df['ChurnRiskScore'] > 0.7) & (df['SubscriptionType'] == 'Premium')]
    st.write(f"Number of high-risk premium customers: {len(high_risk_customers)}")
    st.dataframe(high_risk_customers[['CustomerID', 'TenureMonths', 'MonthlySpending', 'ChurnRiskScore']])

def show_revenue_opportunities(df):
    """
    Display potential revenue opportunities.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    st.subheader("Revenue Opportunities")
    upsell_candidates = df[(df['SubscriptionType'] == 'Standard') & (df['MonthlySpending'] > df['MonthlySpending'].quantile(0.75))]
    st.write(f"Number of potential upsell candidates: {len(upsell_candidates)}")
    st.dataframe(upsell_candidates[['CustomerID', 'TenureMonths', 'MonthlySpending', 'SubscriptionType']])

def show_network_performance(df):
    """
    Display network performance issues.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    st.subheader("Network Performance Issues")
    low_quality_regions = df.groupby('Region')['NetworkQuality'].mean().sort_values().head(3)
    st.write("Regions with lowest average network quality:")
    st.dataframe(low_quality_regions)

def show_churn_prediction(df, models):
    """
    Display the churn prediction interface.

    Args:
        df (pd.DataFrame): Input dataframe
        models (dict): Dictionary containing loaded models
    """
    st.title("🔄 Churn Prediction")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Customer Data Input")

        with st.form("churn_prediction_form"):
            tenure = st.slider("Tenure (months)", 0, 120, 24)
            monthly_spending = st.number_input("Monthly Spending ($)", 0.0, 1000.0, 70.0)
            total_revenue = monthly_spending * tenure
            subscription_type = st.selectbox("Subscription Type", df['SubscriptionType'].unique())
            network_quality = st.slider("Network Quality", 1, 5, 3)
            contract_type = st.selectbox("Contract Type", df['ContractType'].unique())
            payment_method = st.selectbox("Payment Method", df['PaymentMethod'].unique())

            submitted = st.form_submit_button("Predict Churn Risk")

            if submitted:
                input_data = pd.DataFrame({
                    'TenureMonths': [tenure],
                    'MonthlySpending': [monthly_spending],
                    'TotalRevenue': [total_revenue],
                    'SubscriptionType': [subscription_type],
                    'NetworkQuality': [network_quality],
                    'ContractType': [contract_type],
                    'PaymentMethod': [payment_method]
                })

                # Ensure that the input data has all the necessary columns
                input_data = input_data.reindex(columns=df.columns, fill_value=0)

                try:
                    if 'churn' not in models:
                        raise KeyError("Churn model not found in models dictionary")

                    churn_model = models['churn']
                    #st.write(f"Debug: Churn model type: {type(churn_model)}")

                    # Load the scaler and train_columns
                    scaler_path = os.path.join(os.getenv('MODELS_DIR', 'models'), 'scaler.pkl')
                    scaler = joblib.load(scaler_path)
                    train_columns_path = os.path.join(os.getenv('MODELS_DIR', 'models'), 'train_columns.pkl')
                    train_columns = joblib.load(train_columns_path)

                    # Preprocess input data to match training preprocessing (scaling, encoding, etc.)
                    input_data_imputed = input_data.fillna(0)  # Simple imputation (you can add more sophisticated methods)
                    
                    # Encode categorical variables
                    input_data_encoded = pd.get_dummies(input_data_imputed)

                    # Ensure the encoded input data has the same columns as the training data
                    missing_cols = set(train_columns) - set(input_data_encoded.columns)
                    #st.write(f"Debug: Missing columns: {missing_cols}")
                    for col in missing_cols:
                        input_data_encoded[col] = 0
                    input_data_encoded = input_data_encoded.reindex(columns=train_columns, fill_value=0)

                    # Scale input data using the loaded scaler
                    input_data_scaled = scaler.transform(input_data_encoded)

                    # Predict churn probability
                    if hasattr(churn_model, 'predict_proba'):
                        prediction = churn_model.predict_proba(input_data_scaled)[0]
                        churn_probability = prediction[1]
                    elif hasattr(churn_model, 'predict'):
                        prediction = churn_model.predict(input_data_scaled)[0]
                        churn_probability = prediction
                    else:
                        raise AttributeError("Churn model has neither 'predict_proba' nor 'predict' method")

                    with col2:
                        st.subheader("Prediction Results")

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=churn_probability * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Churn Risk"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig)

                        st.markdown("### AI Analysis")
                        explanation = f"Based on the input data, the customer has a {churn_probability:.1%} chance of churning. "
                        explanation += "Factors contributing to this risk include:"
                        if tenure < 12:
                            explanation += "\n- Short tenure"
                        if network_quality < 3:
                            explanation += "\n- Low network quality"
                        if contract_type == "Month-to-month":
                            explanation += "\n- Month-to-month contract"
                        st.info(explanation)

                        st.markdown("### Recommended Actions")
                        if churn_probability < 0.3:
                            st.success("Low Risk: Continue regular engagement")
                        elif churn_probability < 0.7:
                            st.warning("Medium Risk: Consider proactive retention measures")
                        else:
                            st.error("High Risk: Immediate intervention required")

                except KeyError as ke:
                    st.error(f"Model loading error: {str(ke)}")
                except AttributeError as ae:
                    st.error(f"Model method error: {str(ae)}")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.write(f"Debug: models dictionary keys: {list(models.keys())}")
                    st.write(f"Debug: input_data shape: {input_data.shape}")


def show_revenue_forecasting(df, models):
    """
    Display the revenue forecasting interface.
    
    Args:
        df (pd.DataFrame): Input dataframe
        models (dict): Dictionary containing loaded models
    """
    st.title("💰 Revenue Forecasting")

    try:
        historical_data = df.groupby('TenureMonths')['TotalRevenue'].sum().reset_index()
        forecast_periods = st.slider("Forecast Periods (Months)", 1, 24, 12)

        X = historical_data[['TenureMonths']]
        y = historical_data['TotalRevenue']

        model = LinearRegression()
        model.fit(X, y)

        future_months = pd.DataFrame({'TenureMonths': range(X['TenureMonths'].max() + 1, X['TenureMonths'].max() + 1 + forecast_periods)})
        forecast = model.predict(future_months)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=historical_data['TenureMonths'],
            y=historical_data['TotalRevenue'],
            name='Historical Revenue',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=future_months['TenureMonths'],
            y=forecast,
            name='Forecasted Revenue',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title="Revenue Forecast",
            xaxis_title="Tenure (Months)",
            yaxis_title="Revenue ($)",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Forecasted Revenue (Next Period)",
                value=f"${forecast[0]:,.0f}",
                delta=f"{((forecast[0] - y.iloc[-1]) / y.iloc[-1] * 100):+.1f}%"
            )

        with col2:
            accuracy = model.score(X, y) * 100
            st.metric(
                label="Model Accuracy (R²)",
                value=f"{accuracy:.1f}%",
                delta="Based on historical data"
            )

        with col3:
            trend = "Upward" if forecast[-1] > forecast[0] else "Downward"
            st.metric(
                label="Trend",
                value=trend,
                delta=f"{((forecast[-1] - forecast[0]) / forecast[0] * 100):+.1f}%"
            )

    except Exception as e:
        st.error(f"Forecasting failed: {str(e)}")

def show_analytics_dashboard(df):
    """
    Display the analytics dashboard.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    st.title("📊 Analytics Dashboard")
    
    try:
        st.subheader("Revenue by Tenure")
        fig = px.line(
            df.groupby('TenureMonths')['TotalRevenue'].sum().reset_index(),
            x='TenureMonths',
            y='TotalRevenue',
            title='Revenue by Tenure'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Customer Segments")
            fig = px.pie(
                df['SubscriptionType'].value_counts().reset_index(),
                values='count',
                names='SubscriptionType',
                title='Customer Segments Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Service Usage")
            usage_data = pd.DataFrame({
                'Service': ['Data', 'Call', 'SMS'],
                'Usage': [df['DataUsageGB'].mean(), df['CallUsageMin'].mean(), df['SMSUsage'].mean()]
            })
            fig = px.bar(
                usage_data,
                x='Service',
                y='Usage',
                title='Average Service Usage'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Churn Analysis")
        churn_by_contract = df.groupby('ContractType')['Churned'].mean().reset_index()
        fig = px.bar(
            churn_by_contract,
            x='ContractType',
            y='Churned',
            title='Churn Rate by Contract Type'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Customer Satisfaction")
        satisfaction_dist = df['ServiceRating'].value_counts().sort_index().reset_index()
        satisfaction_dist.columns = ['Rating', 'Count']
        fig = px.bar(
            satisfaction_dist,
            x='Rating',
            y='Count',
            title='Distribution of Service Ratings'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to generate analytics dashboard: {str(e)}")

def show_chatbot(df):
    """
    Display a simple rule-based chatbot interface.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    st.title("💬 Analytics Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            response = generate_simple_response(prompt, df)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def generate_simple_response(prompt, df):
    """
    Generate a simple response based on the user's prompt and available data.
    
    Args:
        prompt (str): User's input prompt
        df (pd.DataFrame): Input dataframe
    
    Returns:
        str: Generated response
    """
    prompt = prompt.lower()
    if "customers" in prompt:
        return f"We have {len(df)} customers in our database."
    elif "churn" in prompt:
        churn_rate = (df['Churned'].sum() / len(df)) * 100
        return f"Our current churn rate is {churn_rate:.2f}%."
    elif "revenue" in prompt:
        total_revenue = df['TotalRevenue'].sum()
        return f"Our total revenue is ${total_revenue:,.2f}."
    elif "subscription" in prompt:
        top_subscription = df['SubscriptionType'].mode().values[0]
        return f"Our most common subscription type is {top_subscription}."
    else:
        return "I'm sorry, I don't have enough information to answer that question. Can you please ask about customers, churn, revenue, or subscriptions?"

def show_document_insights():
    """
    Display the document insights interface for analyzing uploaded CSV files.
    """
    st.title("📄 Document Insights")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Overview")
            st.write(user_df.head())
            
            st.subheader("Basic Statistics")
            st.write(user_df.describe())
            
            st.subheader("Column Information")
            for column in user_df.columns:
                st.write(f"**{column}**")
                st.write(f"- Type: {user_df[column].dtype}")
                st.write(f"- Unique Values: {user_df[column].nunique()}")
                st.write(f"- Missing Values: {user_df[column].isnull().sum()}")
            
            st.subheader("Data Visualization")
            column_to_plot = st.selectbox("Select a column to visualize", user_df.columns)
            if user_df[column_to_plot].dtype in ['int64', 'float64']:
                fig = px.histogram(user_df, x=column_to_plot)
                st.plotly_chart(fig)
            else:
                fig = px.bar(user_df[column_to_plot].value_counts().reset_index(), x='index', y=column_to_plot)
                st.plotly_chart(fig)
            
            st.subheader("Custom Analysis")
            custom_query = st.text_area("Enter a custom pandas query (e.g., df['column'].mean())")
            if custom_query:
                try:
                    result = eval(f"user_df.{custom_query}")
                    st.write("Result:", result)
                except Exception as e:
                    st.error(f"Error in custom query: {str(e)}")
            
            st.subheader("AI-Powered Insights")
            if st.button("Generate Insights"):
                insights = generate_document_insights(user_df)
                st.write(insights)
            
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

def generate_document_insights(df):
    """
    Generate insights from the uploaded document.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        str: Generated insights
    """
    num_rows = len(df)
    num_columns = len(df.columns)
    missing_values = df.isnull().sum().sum()
    column_types = df.dtypes.to_dict()
    most_common_values = {col: df[col].mode().values[0] for col in df.columns}

    insights = [
        f"This dataset contains {num_rows} rows and {num_columns} columns, offering a comprehensive view of the data.",
        f"The dataset includes the following columns: {', '.join(df.columns)}. Each of these columns represents a specific data point collected for analysis.",
        f"In total, there are {missing_values} missing values across the dataset, which may require cleaning or further analysis.",
        f"Here are the data types for each column:\n" + "\n".join([f"- {col}: {dtype}" for col, dtype in column_types.items()]),
        f"The most frequent value in each column is:\n" + "\n".join([f"- {col}: {most_common_values[col]}" for col in df.columns]),
    ]

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_columns) > 0:
        avg_values = df[numerical_columns].mean().to_dict()
        insights.append(f"The average values for numerical columns are:\n" + "\n".join([f"- {col}: {avg_values[col]:.2f}" for col in numerical_columns]))

    if len(numerical_columns) > 0:
        trend_analysis = {}
        for col in numerical_columns:
            trend_analysis[col] = "Increasing" if df[col].iloc[-1] > df[col].iloc[0] else "Decreasing"
        insights.append(f"Trends in numerical data:\n" + "\n".join([f"- {col}: {trend_analysis[col]}" for col in numerical_columns]))

    return "\n".join(insights)

def ForeTelAI():
    """
    Main function to run the ForeTel.AI application.
    """
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Home", "Churn Prediction", "Revenue Forecasting", "Analytics Dashboard", "Chat Assistant", "Document Insights"],
            icons=['house', 'person-x', 'cash-coin', 'graph-up', 'chat-dots', 'file-text'],
            menu_icon="cast",
            default_index=0,
        )
    
    if selected == "Home":
        show_home_page(df)
    elif selected == "Churn Prediction":
        show_churn_prediction(df, models)
    elif selected == "Revenue Forecasting":
        show_revenue_forecasting(df, models)
    elif selected == "Analytics Dashboard":
        show_analytics_dashboard(df)
    elif selected == "Chat Assistant":
        show_chatbot(df)
    elif selected == "Document Insights":
        show_document_insights()

if __name__ == "__main__":
    ForeTelAI()
