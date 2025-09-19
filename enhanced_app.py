"""
Enhanced ForeTel.AI Application with Modern UI/UX and AI Chatbot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import json
from datetime import datetime, timedelta
import time
import random

# Import custom modules
from enhanced_config import ENHANCED_UI_CSS, ENHANCED_JS, CHATBOT_CONFIG, FEATURE_FLAGS
from chatbot_engine import get_chatbot, render_chat_message, render_suggestion_chips
from churn_predictor import ChurnPredictor
from revenue_forecasting_model import RevenueForecaster

# Page Configuration
st.set_page_config(
    page_title="ForeTel.AI - Enterprise Telecom Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Enhanced Styling
st.markdown(ENHANCED_UI_CSS, unsafe_allow_html=True)
st.markdown(ENHANCED_JS, unsafe_allow_html=True)

# Initialize Session State
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}

# Load Data and Models
@st.cache_data
def load_data():
    """Load telecom dataset"""
    try:
        if os.path.exists('telecom_dataset_generated.csv'):
            df = pd.read_csv('telecom_dataset_generated.csv')
        elif os.path.exists('telecom_dataset.csv'):
            df = pd.read_csv('telecom_dataset.csv')
        else:
            st.error("Dataset not found!")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    try:
        if os.path.exists('models/enhanced_churn_model.pkl'):
            models['churn'] = joblib.load('models/enhanced_churn_model.pkl')
        if os.path.exists('models/revenue_forecasting.pkl'):
            models['revenue'] = joblib.load('models/revenue_forecasting.pkl')
        if os.path.exists('models/scaler.pkl'):
            models['scaler'] = joblib.load('models/scaler.pkl')
        if os.path.exists('models/imputer.pkl'):
            models['imputer'] = joblib.load('models/imputer.pkl')
        if os.path.exists('models/train_columns.pkl'):
            models['columns'] = joblib.load('models/train_columns.pkl')
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

# Animated Header
def render_animated_header():
    """Render animated header with particles"""
    st.markdown("""
    <div class="main-header">
        <div style="text-align: center;">
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                       font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem;">
                📡 ForeTel.AI
            </h1>
            <p style="font-size: 1.2rem; color: var(--text-secondary); margin-bottom: 1rem;">
                Enterprise Telecom Analytics Platform
            </p>
            <div class="status-indicator">
                <span class="status-online">●</span> System Online | 
                <span style="color: var(--text-secondary);">Last Updated: {}</span>
            </div>
        </div>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

# Enhanced Navigation
def render_navigation():
    """Render enhanced navigation sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: var(--text-primary);">Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation items with icons
        nav_items = [
            ("🏠", "Home", "home"),
            ("📊", "Churn Prediction", "churn"),
            ("💰", "Revenue Forecasting", "revenue"),
            ("📈", "Analytics Dashboard", "analytics"),
            ("🤖", "AI Assistant", "chat"),
            ("📄", "Document Insights", "documents"),
            ("⚙️", "Settings", "settings")
        ]
        
        for icon, label, key in nav_items:
            if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.current_page = label
                st.rerun()
        
        # Theme Toggle
        st.markdown("---")
        if st.button("🌙 Toggle Theme", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

# Enhanced Metrics Display
def render_metrics_cards(df):
    """Render animated metrics cards"""
    if df is None:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Total Customers", len(df), "👥", "#667eea"),
        ("Churn Rate", f"{(df.get('Churned', df.get('churn', [0])).mean() * 100):.1f}%", "📉", "#ff6b6b"),
        ("Avg Revenue", f"₹{df.get('TotalRevenue', df.get('MonthlySpending', [0])).mean():.0f}", "💰", "#00d4aa"),
        ("Active Services", len(df.columns), "🔧", "#feca57")
    ]
    
    for i, (col, (title, value, icon, color)) in enumerate(zip([col1, col2, col3, col4], metrics)):
        with col:
            st.markdown(f"""
            <div class="metric-card interactive" style="border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="color: {color}; margin: 0; font-size: 0.9rem;">{title}</h3>
                        <p style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0; color: var(--text-primary);">{value}</p>
                    </div>
                    <div style="font-size: 2rem; opacity: 0.7;">{icon}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Home Page
def render_home_page():
    """Render enhanced home page"""
    render_animated_header()
    
    df = load_data()
    render_metrics_cards(df)
    
    # Feature Cards
    st.markdown("## 🚀 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        {
            "title": "Churn Prediction",
            "description": "Advanced ML models to predict customer churn with 95%+ accuracy",
            "icon": "📊",
            "color": "#667eea"
        },
        {
            "title": "Revenue Forecasting", 
            "description": "Ensemble models for accurate revenue predictions and business planning",
            "icon": "💰",
            "color": "#764ba2"
        },
        {
            "title": "Real-time Analytics",
            "description": "Interactive dashboards with live data insights and visualizations",
            "icon": "📈",
            "color": "#4facfe"
        }
    ]
    
    for i, (col, feature) in enumerate(zip([col1, col2, col3], features)):
        with col:
            st.markdown(f"""
            <div class="feature-card interactive" onclick="window.location.reload()">
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">{feature['icon']}</div>
                    <h3 style="color: {feature['color']}; margin-bottom: 0.5rem;">{feature['title']}</h3>
                    <p style="color: var(--text-secondary); line-height: 1.5;">{feature['description']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent Activity
    st.markdown("## 📋 Recent Activity")
    
    activities = [
        ("Model Training Completed", "Churn prediction model retrained with latest data", "2 hours ago", "success"),
        ("Data Pipeline Updated", "New customer data processed successfully", "4 hours ago", "info"),
        ("Alert Generated", "High churn risk detected in premium segment", "6 hours ago", "warning")
    ]
    
    for activity, description, time_ago, status in activities:
        color = {"success": "#00d4aa", "info": "#4facfe", "warning": "#feca57"}[status]
        st.markdown(f"""
        <div class="info-card" style="border-left: 4px solid {color}; margin: 0.5rem 0;">
            <div style="display: flex; justify-content: between; align-items: center;">
                <div style="flex: 1;">
                    <h4 style="margin: 0; color: var(--text-primary);">{activity}</h4>
                    <p style="margin: 0.25rem 0 0 0; color: var(--text-secondary); font-size: 0.9rem;">{description}</p>
                </div>
                <div style="color: var(--text-secondary); font-size: 0.8rem;">{time_ago}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Churn Prediction Page
def render_churn_prediction():
    """Render enhanced churn prediction page"""
    st.markdown("# 📊 Churn Prediction")
    
    df = load_data()
    models = load_models()
    
    if df is None or 'churn' not in models:
        st.error("Data or model not available!")
        return
    
    # Input Section
    st.markdown("## 🎯 Customer Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Customer Details")
        
        # Customer input form
        age = st.slider("Age", 18, 80, 35)
        monthly_income = st.number_input("Monthly Income (₹)", 10000, 200000, 50000)
        data_usage = st.slider("Data Usage (GB)", 0.0, 50.0, 15.0)
        call_usage = st.slider("Call Usage (Minutes)", 0, 1000, 250)
        monthly_spending = st.number_input("Monthly Spending (₹)", 100, 5000, 500)
        
        # Predict button
        if st.button("🔮 Predict Churn Risk", use_container_width=True):
            # Create prediction data
            prediction_data = pd.DataFrame({
                'Age': [age],
                'MonthlyIncome': [monthly_income],
                'DataUsageGB': [data_usage],
                'CallUsageMin': [call_usage],
                'MonthlySpending': [monthly_spending]
            })
            
            try:
                # Make prediction
                churn_prob = models['churn'].predict_proba(prediction_data)[0][1]
                
                # Store result in session state
                st.session_state.prediction_result = {
                    'churn_probability': churn_prob,
                    'risk_level': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.4 else 'Low'
                }
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    with col2:
        st.markdown("### Prediction Results")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            prob = result['churn_probability']
            risk = result['risk_level']
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk %"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'family': "Inter"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            color = {"High": "#ff6b6b", "Medium": "#feca57", "Low": "#00d4aa"}[risk]
            st.markdown(f"""
            <div class="info-card" style="border: 2px solid {color}; text-align: center;">
                <h3 style="color: {color}; margin-bottom: 0.5rem;">Risk Level: {risk}</h3>
                <p style="font-size: 1.1rem;">Churn Probability: {prob:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Enter customer details and click 'Predict Churn Risk' to see results")
    
    # Churn Analysis Dashboard
    st.markdown("## 📈 Churn Analysis Dashboard")
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by age group
        if 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
            churn_by_age = df.groupby('AgeGroup')['Churned'].mean() if 'Churned' in df.columns else df.groupby('AgeGroup').size()
            
            fig = px.bar(
                x=churn_by_age.index, 
                y=churn_by_age.values,
                title="Churn Rate by Age Group",
                color=churn_by_age.values,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly spending distribution
        if 'MonthlySpending' in df.columns:
            fig = px.histogram(
                df, 
                x='MonthlySpending',
                title="Monthly Spending Distribution",
                nbins=30,
                color_discrete_sequence=["#667eea"]
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"}
            )
            st.plotly_chart(fig, use_container_width=True)

# Enhanced AI Chat Page
def render_ai_chat():
    """Render enhanced AI chat interface"""
    st.markdown("# 🤖 AI Assistant")
    
    # Get chatbot instance
    chatbot = get_chatbot()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown("""
        <div class="chat-container">
        """, unsafe_allow_html=True)
        
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                render_chat_message(message['user'], is_user=True)
                render_chat_message(message['bot'], is_user=False)
        else:
            # Welcome message
            welcome_msg = "Hello! I'm your ForeTel.AI assistant. How can I help you with telecom analytics today?"
            render_chat_message(welcome_msg, is_user=False)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message...", 
            key="chat_input",
            placeholder="Ask about churn prediction, revenue forecasting, or data insights..."
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    # Process chat input
    if send_button and user_input:
        # Show typing indicator
        with st.spinner("AI is thinking..."):
            time.sleep(0.5)  # Simulate processing time
            response = chatbot.get_response(user_input)
        
        # Add to chat history
        st.session_state.chat_history.append({
            'user': user_input,
            'bot': response
        })
        
        st.rerun()
    
    # Suggestion chips
    st.markdown("---")
    suggestions = chatbot.get_suggestions()
    selected_suggestion = render_suggestion_chips(suggestions, "chat")
    
    if selected_suggestion:
        # Process suggestion
        response = chatbot.get_response(selected_suggestion)
        st.session_state.chat_history.append({
            'user': selected_suggestion,
            'bot': response
        })
        st.rerun()
    
    # Chat controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            chatbot.clear_history()
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            chat_export = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                "Download Chat History",
                chat_export,
                "chat_history.json",
                "application/json"
            )
    
    with col3:
        if st.button("🔄 Restart AI", use_container_width=True):
            # Reinitialize chatbot
            st.cache_resource.clear()
            st.rerun()

# Enhanced Analytics Dashboard
def render_analytics_dashboard():
    """Render enhanced analytics dashboard"""
    st.markdown("# 📈 Analytics Dashboard")
    
    df = load_data()
    if df is None:
        st.error("Data not available!")
        return
    
    # Dashboard metrics
    render_metrics_cards(df)
    
    # Interactive charts
    st.markdown("## 📊 Interactive Visualizations")
    
    # Chart selection
    chart_type = st.selectbox(
        "Select Visualization",
        ["Customer Distribution", "Revenue Analysis", "Usage Patterns", "Correlation Matrix"]
    )
    
    if chart_type == "Customer Distribution":
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Gender' in df.columns:
                fig = px.pie(df, names='Gender', title="Gender Distribution")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Region' in df.columns:
                fig = px.bar(df['Region'].value_counts(), title="Customers by Region")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Revenue Analysis":
        if 'TotalRevenue' in df.columns or 'MonthlySpending' in df.columns:
            revenue_col = 'TotalRevenue' if 'TotalRevenue' in df.columns else 'MonthlySpending'
            
            fig = px.histogram(df, x=revenue_col, title=f"{revenue_col} Distribution")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Usage Patterns":
        usage_cols = [col for col in df.columns if 'Usage' in col or 'Data' in col]
        if usage_cols:
            selected_col = st.selectbox("Select Usage Metric", usage_cols)
            fig = px.box(df, y=selected_col, title=f"{selected_col} Distribution")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Correlation Matrix":
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Limit to 10 columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title="Feature Correlation Matrix")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True)

# Main Application
def main():
    """Main application function"""
    # Render navigation
    render_navigation()
    
    # Route to appropriate page
    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Churn Prediction":
        render_churn_prediction()
    elif st.session_state.current_page == "AI Assistant":
        render_ai_chat()
    elif st.session_state.current_page == "Analytics Dashboard":
        render_analytics_dashboard()
    else:
        st.info(f"Page '{st.session_state.current_page}' is under development!")

if __name__ == "__main__":
    main()
