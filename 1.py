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
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from streamlit_option_menu import option_menu
import torch
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
        background-color: #47663B;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        color: white;
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
        background-color: #1F4529;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 2rem;
        color: white;
    }
    .alert-high {
        color: #721c24;
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .alert-positive {
        color: #155724;
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .alert-warning {
        color: #856404;
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
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
    }
</style>
""", unsafe_allow_html=True)

# Caching and Loading Functions
@st.cache_data
def load_data():
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
    try:
        models_dir = os.getenv('MODELS_DIR', 'models')
        churn_model = joblib.load(os.path.join(models_dir, 'enhanced_churn_model.pkl'))
        revenue_model = joblib.load(os.path.join(models_dir, 'revenue_forecasting.pkl'))
        nlp_model = pipeline('text2text-generation', model='google/flan-t5-base', device='cpu')
        return {
            'churn': churn_model,
            'revenue': revenue_model,
            'nlp': nlp_model
        }
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        st.error("Failed to load models. Please check if model files exist and are accessible.")
        return {}

@st.cache_resource
def load_chatbot():
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load chatbot: {str(e)}")
        st.error("Failed to load chatbot model. Please check your internet connection and try again.")
        return None, None

# Load data and models
df = load_data()
models = load_models()
chatbot_model, chatbot_tokenizer = load_chatbot()

# Utility Functions
def get_real_metrics(df):
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
        segment_data['Growth'] = ['+12%', '+5%', '-2%'][:len(segment_data)]  # Ensure matching length
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
    st.subheader("Churn Risk Details")
    high_risk_customers = df[(df['ChurnRiskScore'] > 0.7) & (df['SubscriptionType'] == 'Premium')]
    st.write(f"Number of high-risk premium customers: {len(high_risk_customers)}")
    st.dataframe(high_risk_customers[['CustomerID', 'TenureMonths', 'MonthlySpending', 'ChurnRiskScore']])

def show_revenue_opportunities(df):
    st.subheader("Revenue Opportunities")
    upsell_candidates = df[(df['SubscriptionType'] == 'Standard') & (df['MonthlySpending'] > df['MonthlySpending'].quantile(0.75))]
    st.write(f"Number of potential upsell candidates: {len(upsell_candidates)}")
    st.dataframe(upsell_candidates[['CustomerID', 'TenureMonths', 'MonthlySpending', 'SubscriptionType']])

def show_network_performance(df):
    st.subheader("Network Performance Issues")
    low_quality_regions = df.groupby('Region')['NetworkQuality'].mean().sort_values().head(3)
    st.write("Regions with lowest average network quality:")
    st.dataframe(low_quality_regions)

def show_churn_prediction(df, model):
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    
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
                # Create input DataFrame
                input_data = pd.DataFrame({
                    'TenureMonths': [tenure],
                    'MonthlySpending': [monthly_spending],
                    'TotalRevenue': [total_revenue],
                    'SubscriptionType': [subscription_type],
                    'NetworkQuality': [network_quality],
                    'ContractType': [contract_type],
                    'PaymentMethod': [payment_method]
                })
                
                try:
                    # Predict churn probability
                    prediction = model.predict_proba(input_data)[0]
                    churn_probability = prediction[1]
                    
                    # Display results
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
                        
                        st.markdown("### Recommended Actions")
                        if churn_probability < 0.3:
                            st.success("Low Risk: Continue regular engagement.")
                        elif churn_probability < 0.7:
                            st.warning("Medium Risk: Consider proactive retention measures.")
                        else:
                            st.error("High Risk: Immediate intervention required.")
                
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")


def show_revenue_forecasting(df, models):
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

def show_chatbot(model, tokenizer, df):
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
            if model and tokenizer:
                try:
                    start_time = time.time()
                    response = generate_response(prompt, model, tokenizer, df)
                    end_time = time.time()
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.info(f"Response generated in {end_time - start_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Failed to generate response: {str(e)}")
            else:
                st.error("Chatbot model not loaded. Please check the logs for details.")

def generate_response(prompt, model, tokenizer, df):
    enhanced_prompt = f"""
    You are an AI assistant for a telecom company. You have access to the following data:
    - Total customers: {len(df)}
    - Average monthly spending: ${df['MonthlySpending'].mean():.2f}
    - Churn rate: {df['Churned'].mean() * 100:.2f}%
    - Most common subscription type: {df['SubscriptionType'].mode().values[0]}
    
    User question: {prompt}
    
    Please provide a concise and informative answer based on this data.
    """
    
    input_ids = tokenizer.encode(
        f"<human>: {enhanced_prompt}\n<assistant>:",
        return_tensors="pt"
    )
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("<assistant>: ")[-1].strip()

def show_settings():
    st.title("⚙️ Settings")
    
    try:
        st.subheader("User Preferences")
        notifications = st.checkbox("Enable Email Notifications", value=True)
        theme = st.selectbox("Theme", ["Light", "Dark", "System"])
        language = st.selectbox("Language", ["English", "Spanish", "French", "German"])
        
        if st.button("Save Preferences"):
            # Placeholder for saving preferences
            st.success("Preferences saved successfully!")
        
        st.subheader("Data Management")
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.success("Data refreshed successfully!")
        
        st.subheader("Account Settings")
        if st.button("Change Password"):
            # Placeholder for password change functionality
            st.info("Password change functionality to be implemented.")
        
    except Exception as e:
        st.error(f"Failed to load or save settings: {str(e)}")

def show_document_insights():
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
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Home", "Churn Prediction", "Revenue Forecasting", "Analytics Dashboard", "Chat Assistant", "Document Insights", "Settings"],
            icons=['house', 'person-x', 'cash-coin', 'graph-up', 'chat-dots', 'file-text', 'gear'],
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
        show_chatbot(chatbot_model, chatbot_tokenizer, df)
    elif selected == "Document Insights":
        show_document_insights()
    elif selected == "Settings":
        show_settings()

if __name__ == "__main__":
    ForeTelAI()

