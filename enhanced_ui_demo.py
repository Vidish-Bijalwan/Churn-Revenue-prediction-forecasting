"""
ForeTel.AI UI/UX Demo with Modern Animations and AI Chatbot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import random
import json

# Import chatbot engine - with fallback for deployment
try:
    from chatbot_engine import ChatbotEngine
    chatbot_available = True
except ImportError:
    chatbot_available = False
    # Simple fallback chatbot class
    class ChatbotEngine:
        def __init__(self):
            pass
        
        def get_response(self, message):
            responses = [
                "I'm here to help with your telecom analytics questions!",
                "Based on your data, I can provide insights about customer churn and revenue forecasting.",
                "Let me analyze that for you. What specific metrics would you like to explore?",
                "I can help you understand customer behavior patterns and revenue trends.",
                "Would you like me to explain the churn prediction results or revenue forecasts?"
            ]
            return random.choice(responses)
        
        def get_suggestions(self):
            return [
                "What's my churn rate?",
                "Show revenue forecast",
                "Customer insights",
                "Risk analysis"
            ]

# Import config - with fallback for Streamlit Cloud
try:
    from enhanced_config import ENHANCED_UI_CSS, ENHANCED_JS, CHATBOT_CONFIG, FEATURE_FLAGS
except ImportError:
    # Fallback CSS for Streamlit Cloud deployment
    UI_CSS = """
    <style>
    .metric-card, .info-card, .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .metric-card:hover, .info-card:hover, .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(31, 38, 135, 0.2);
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 12px;
    }
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .chat-message.bot {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        margin-right: 20%;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """
    JS_CODE = "<script></script>"
    CHATBOT_CONFIG = {}
    FEATURE_FLAGS = {"modern_ui": True}

# Page Configuration
st.set_page_config(
    page_title="ForeTel.AI - Modern UI Demo",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Modern Styling
if 'ENHANCED_UI_CSS' in locals():
    st.markdown(ENHANCED_UI_CSS, unsafe_allow_html=True)
    st.markdown(ENHANCED_JS, unsafe_allow_html=True)
else:
    st.markdown(UI_CSS, unsafe_allow_html=True)
    st.markdown(JS_CODE, unsafe_allow_html=True)

# Initialize Session State
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"
if 'user_settings' not in st.session_state:
    st.session_state.user_settings = {
        'notifications': True,
        'auto_refresh': True,
        'chart_animations': True,
        'sound_effects': False,
        'language': 'English',
        'timezone': 'Asia/Kolkata'
    }

# Simple Chatbot Class
class SimpleChatbot:
    def __init__(self):
        self.responses = {
            "hello": ["Hello! Welcome to ForeTel.AI! How can I help you today?", "Hi there! I'm your AI assistant for telecom analytics."],
            "churn": ["Churn prediction helps identify customers at risk of leaving. Our models analyze usage patterns and behavior.", "Customer churn analysis uses machine learning to predict which customers might cancel their service."],
            "revenue": ["Revenue forecasting predicts future income using historical data and market trends.", "Our revenue models help you plan business strategies and budget allocation."],
            "help": ["I can help with churn prediction, revenue forecasting, and data analytics questions.", "Try asking about customer analysis, predictions, or platform features."],
            "default": ["That's interesting! I specialize in telecom analytics. Ask me about churn prediction or revenue forecasting.", "I'm here to help with ForeTel.AI features. What would you like to know?"]
        }
    
    def get_response(self, user_input):
        user_input = user_input.lower()
        for key, responses in self.responses.items():
            if key in user_input:
                return random.choice(responses)
        return random.choice(self.responses["default"])

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    return SimpleChatbot()

chatbot = get_chatbot()

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_customers = 1000
    
    data = {
        'CustomerID': range(1, n_customers + 1),
        'Age': np.random.randint(18, 80, n_customers),
        'Gender': np.random.choice(['Male', 'Female'], n_customers),
        'MonthlyIncome': np.random.normal(50000, 15000, n_customers),
        'DataUsageGB': np.random.exponential(15, n_customers),
        'CallUsageMin': np.random.normal(250, 100, n_customers),
        'MonthlySpending': np.random.normal(500, 150, n_customers),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_customers),
        'Churned': np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

# Animated Header
def render_animated_header():
    st.markdown("""
    <div class="main-header">
        <div style="text-align: center;">
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                       font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem;">
                📡 ForeTel.AI
            </h1>
            <p style="font-size: 1.2rem; color: var(--text-secondary); margin-bottom: 1rem;">
                Modern UI/UX with AI-Powered Analytics
            </p>
            <div class="status-indicator">
                <span class="status-online">●</span> Modern UI Active | 
                <span style="color: var(--text-secondary);">Last Updated: {}</span>
            </div>
        </div>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

# Navigation
def render_navigation():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: var(--text-primary);">🚀 Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        nav_items = [
            ("🏠", "Home", "home"),
            ("📊", "Churn Prediction", "churn"),
            ("💰", "Revenue Forecasting", "revenue"),
            ("📈", "Analytics Dashboard", "analytics"),
            ("🤖", "AI Chat Assistant", "chat"),
            ("📄", "Reports & Insights", "reports"),
            ("⚙️", "Settings", "settings")
        ]
        
        for icon, label, key in nav_items:
            if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.current_page = label
                st.rerun()
        
        st.markdown("---")
        if st.button("🌙 Toggle Theme", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

# Metrics Display
def render_metrics_cards(df):
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Total Customers", len(df), "👥", "#667eea"),
        ("Churn Rate", f"{df['Churned'].mean() * 100:.1f}%", "📉", "#ff6b6b"),
        ("Avg Revenue", f"₹{df['MonthlySpending'].mean():.0f}", "💰", "#00d4aa"),
        ("Data Usage", f"{df['DataUsageGB'].mean():.1f} GB", "📱", "#feca57")
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
    render_animated_header()
    
    df = generate_sample_data()
    render_metrics_cards(df)
    
    st.markdown("## 🚀 Features Demo")
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        {
            "title": "Modern UI/UX",
            "description": "Sleek animations, smooth transitions, and interactive elements",
            "icon": "🎨",
            "color": "#667eea"
        },
        {
            "title": "AI-Powered Chat", 
            "description": "Intelligent chatbot for analytics queries and assistance",
            "icon": "🤖",
            "color": "#764ba2"
        },
        {
            "title": "Real-time Analytics",
            "description": "Interactive dashboards with live data visualizations",
            "icon": "📈",
            "color": "#4facfe"
        }
    ]
    
    for i, (col, feature) in enumerate(zip([col1, col2, col3], features)):
        with col:
            st.markdown(f"""
            <div class="feature-card interactive">
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">{feature['icon']}</div>
                    <h3 style="color: {feature['color']}; margin-bottom: 0.5rem;">{feature['title']}</h3>
                    <p style="color: var(--text-secondary); line-height: 1.5;">{feature['description']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Analytics Demo
def render_analytics_demo():
    st.markdown("# 📊 Analytics Demo")
    
    df = generate_sample_data()
    render_metrics_cards(df)
    
    st.markdown("## 📈 Interactive Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = px.histogram(df, x='Age', title="Customer Age Distribution", 
                          color_discrete_sequence=["#667eea"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn by region
        churn_by_region = df.groupby('Region')['Churned'].mean()
        fig = px.bar(x=churn_by_region.index, y=churn_by_region.values,
                    title="Churn Rate by Region", color=churn_by_region.values,
                    color_continuous_scale="Viridis")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Usage patterns
    st.markdown("### 📱 Usage Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='DataUsageGB', y='MonthlySpending', 
                        color='Churned', title="Data Usage vs Spending")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='Gender', y='CallUsageMin', title="Call Usage by Gender")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        st.plotly_chart(fig, use_container_width=True)

# AI Chat Demo
def render_chat_demo():
    st.markdown("# 🤖 AI Assistant Demo")
    
    st.markdown("""
    <div class="info-card">
        <h3>💡 AI Chat Features</h3>
        <p>This demo showcases the chatbot with modern UI and intelligent responses.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                # User message
                st.markdown(f"""
                <div class="chat-message user">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.2rem;">🧑‍💼</span>
                        <div>{message['user']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Bot message
                st.markdown(f"""
                <div class="chat-message bot">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.2rem;">🤖</span>
                        <div>{message['bot']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="chat-message bot">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">🤖</span>
                    <div>Hello! I'm your ForeTel.AI assistant. How can I help you today?</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message...", 
            key="chat_input",
            placeholder="Ask about churn prediction, revenue forecasting, or analytics..."
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    # Process input
    if send_button and user_input:
        response = chatbot.get_response(user_input)
        st.session_state.chat_history.append({
            'user': user_input,
            'bot': response
        })
        st.rerun()
    
    # Suggestion chips
    st.markdown("---")
    st.markdown("💡 **Quick Questions:**")
    
    suggestions = [
        "Hello, how are you?",
        "Tell me about churn prediction",
        "How does revenue forecasting work?",
        "I need help with analytics"
    ]
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        col = cols[i % 2]
        if col.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            response = chatbot.get_response(suggestion)
            st.session_state.chat_history.append({
                'user': suggestion,
                'bot': response
            })
            st.rerun()
    
    # Chat controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
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

# Churn Prediction Page
def render_churn_prediction():
    st.markdown("# 📊 Customer Churn Prediction")
    
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Predict Customer Churn Risk</h3>
        <p>Use our advanced ML model to identify customers at risk of churning and take proactive retention actions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Customer Information")
        
        # Customer input form
        with st.form("churn_prediction_form"):
            age = st.slider("Age", 18, 80, 35, help="Customer's age")
            gender = st.selectbox("Gender", ["Male", "Female"])
            monthly_income = st.number_input("Monthly Income (₹)", 10000, 200000, 50000, step=1000)
            region = st.selectbox("Region", ["North", "South", "East", "West"])
            
            st.markdown("#### 📱 Usage Patterns")
            data_usage = st.slider("Data Usage (GB/month)", 0.0, 50.0, 15.0, step=0.5)
            call_usage = st.slider("Call Usage (Minutes/month)", 0, 1000, 250, step=10)
            sms_usage = st.slider("SMS Usage (count/month)", 0, 500, 100, step=5)
            
            st.markdown("#### 💳 Financial Information")
            monthly_spending = st.number_input("Monthly Spending (₹)", 100, 5000, 500, step=50)
            payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Bank Transfer", "Cash"])
            
            st.markdown("#### 📞 Service Quality")
            network_quality = st.slider("Network Quality (1-10)", 1, 10, 7)
            service_rating = st.slider("Service Rating (1-10)", 1, 10, 8)
            customer_care_calls = st.number_input("Customer Care Calls (last 6 months)", 0, 20, 2)
            
            predict_button = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)
        
        if predict_button:
            # Simulate prediction
            risk_factors = []
            if monthly_spending < 300: risk_factors.append("Low spending")
            if customer_care_calls > 5: risk_factors.append("High support calls")
            if network_quality < 5: risk_factors.append("Poor network quality")
            if service_rating < 6: risk_factors.append("Low satisfaction")
            
            churn_prob = min(0.9, len(risk_factors) * 0.2 + random.uniform(0.1, 0.3))
            
            st.session_state.churn_result = {
                'probability': churn_prob,
                'risk_level': 'High' if churn_prob > 0.6 else 'Medium' if churn_prob > 0.3 else 'Low',
                'risk_factors': risk_factors
            }
    
    with col2:
        st.markdown("### 📈 Prediction Results")
        
        if 'churn_result' in st.session_state:
            result = st.session_state.churn_result
            prob = result['probability']
            risk = result['risk_level']
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk %", 'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': 'white'},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(0, 212, 170, 0.3)"},
                        {'range': [30, 60], 'color': "rgba(254, 202, 87, 0.3)"},
                        {'range': [60, 100], 'color': "rgba(255, 107, 107, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'family': "Inter"},
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            color = {"High": "#ff6b6b", "Medium": "#feca57", "Low": "#00d4aa"}[risk]
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {color}; text-align: center;">
                <h3 style="color: {color}; margin-bottom: 0.5rem;">Risk Level: {risk}</h3>
                <p style="font-size: 1.2rem; margin: 0;">Churn Probability: {prob:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk factors
            if result['risk_factors']:
                st.markdown("#### ⚠️ Risk Factors Identified:")
                for factor in result['risk_factors']:
                    st.markdown(f"• {factor}")
            
            # Recommendations
            st.markdown("#### 💡 Recommended Actions:")
            if risk == "High":
                recommendations = [
                    "Immediate contact by retention team",
                    "Offer personalized discount or upgrade",
                    "Schedule customer satisfaction survey",
                    "Assign dedicated account manager"
                ]
            elif risk == "Medium":
                recommendations = [
                    "Send targeted retention offers",
                    "Improve service quality monitoring",
                    "Proactive customer engagement"
                ]
            else:
                recommendations = [
                    "Continue regular service monitoring",
                    "Maintain current service quality",
                    "Consider upselling opportunities"
                ]
            
            for rec in recommendations:
                st.markdown(f"✓ {rec}")
        else:
            st.info("👆 Fill out the customer information form and click 'Predict Churn Risk' to see results")

# Revenue Forecasting Page
def render_revenue_forecasting():
    st.markdown("# 💰 Revenue Forecasting")
    
    st.markdown("""
    <div class="info-card">
        <h3>📈 Predict Future Revenue</h3>
        <p>Forecast revenue using advanced time series analysis and customer behavior patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Forecasting Parameters")
        
        with st.form("revenue_forecast_form"):
            forecast_period = st.selectbox("Forecast Period", ["1 Month", "3 Months", "6 Months", "1 Year"])
            customer_segment = st.selectbox("Customer Segment", ["All Customers", "Premium", "Standard", "Basic"])
            
            st.markdown("#### 📈 Current Metrics")
            current_customers = st.number_input("Current Active Customers", 1000, 100000, 10000, step=100)
            avg_monthly_revenue = st.number_input("Average Monthly Revenue per Customer (₹)", 100, 2000, 500, step=50)
            churn_rate = st.slider("Current Churn Rate (%)", 0.0, 50.0, 15.0, step=0.5)
            
            st.markdown("#### 🎯 Growth Assumptions")
            new_customer_rate = st.slider("New Customer Acquisition Rate (%/month)", 0.0, 20.0, 5.0, step=0.5)
            price_increase = st.slider("Expected Price Increase (%)", 0.0, 20.0, 3.0, step=0.5)
            seasonal_factor = st.selectbox("Seasonal Factor", ["None", "Holiday Boost", "Summer Dip", "Year-end Surge"])
            
            forecast_button = st.form_submit_button("📈 Generate Forecast", use_container_width=True)
        
        if forecast_button:
            # Simulate revenue forecasting
            periods = {"1 Month": 1, "3 Months": 3, "6 Months": 6, "1 Year": 12}[forecast_period]
            
            monthly_revenues = []
            customers = current_customers
            monthly_rev = avg_monthly_revenue
            
            for month in range(periods + 1):
                if month > 0:
                    # Apply growth and churn
                    new_customers = customers * (new_customer_rate / 100)
                    churned_customers = customers * (churn_rate / 100)
                    customers = customers + new_customers - churned_customers
                    
                    # Apply price changes
                    monthly_rev = monthly_rev * (1 + price_increase / 100 / 12)
                
                # Apply seasonal factors
                seasonal_multiplier = 1.0
                if seasonal_factor == "Holiday Boost" and month % 12 in [11, 0]:
                    seasonal_multiplier = 1.2
                elif seasonal_factor == "Summer Dip" and month % 12 in [5, 6, 7]:
                    seasonal_multiplier = 0.9
                elif seasonal_factor == "Year-end Surge" and month % 12 == 11:
                    seasonal_multiplier = 1.15
                
                total_revenue = customers * monthly_rev * seasonal_multiplier
                monthly_revenues.append(total_revenue)
            
            st.session_state.revenue_forecast = {
                'revenues': monthly_revenues,
                'periods': periods,
                'final_customers': customers,
                'growth_rate': ((monthly_revenues[-1] / monthly_revenues[0]) - 1) * 100 if monthly_revenues[0] > 0 else 0
            }
    
    with col2:
        st.markdown("### 📈 Forecast Results")
        
        if 'revenue_forecast' in st.session_state:
            forecast = st.session_state.revenue_forecast
            
            # Revenue trend chart
            months = list(range(len(forecast['revenues'])))
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=months,
                y=forecast['revenues'],
                mode='lines+markers',
                name='Forecasted Revenue',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Revenue Forecast Trend",
                xaxis_title="Months",
                yaxis_title="Revenue (₹)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #00d4aa;">
                    <h4 style="color: #00d4aa; margin: 0;">Total Forecast</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">₹{sum(forecast['revenues'][1:]):.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2b:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #667eea;">
                    <h4 style="color: #667eea; margin: 0;">Growth Rate</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{forecast['growth_rate']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Key insights
            st.markdown("#### 💡 Key Insights:")
            insights = [
                f"Expected {forecast['growth_rate']:.1f}% revenue growth over {forecast['periods']} months",
                f"Projected customer base: {forecast['final_customers']:.0f} customers",
                f"Average monthly revenue: ₹{sum(forecast['revenues'][1:]) / len(forecast['revenues'][1:]):.0f}",
            ]
            
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.info("👆 Set your forecasting parameters and click 'Generate Forecast' to see predictions")

# Reports & Insights Page
def render_reports():
    st.markdown("# 📄 Reports & Insights")
    
    df = generate_sample_data()
    
    # Executive Summary
    st.markdown("## 📋 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #667eea;">
            <h4 style="color: #667eea;">Customer Health Score</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #667eea;">8.2/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #00d4aa;">
            <h4 style="color: #00d4aa;">Revenue Growth</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #00d4aa;">+12.5%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #feca57;">
            <h4 style="color: #feca57;">Retention Rate</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #feca57;">87.3%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #ff6b6b;">
            <h4 style="color: #ff6b6b;">At-Risk Customers</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #ff6b6b;">156</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Analytics
    st.markdown("## 📊 Detailed Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer segmentation
        segments = ['Premium', 'Standard', 'Basic']
        segment_counts = [250, 500, 250]
        
        fig = go.Figure(data=[go.Pie(
            labels=segments,
            values=segment_counts,
            hole=0.4,
            marker_colors=['#667eea', '#00d4aa', '#feca57']
        )])
        
        fig.update_layout(
            title="Customer Segmentation",
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly trends
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        revenue_trend = [450000, 465000, 480000, 495000, 510000, 525000]
        churn_trend = [15.2, 14.8, 14.5, 14.1, 13.8, 13.5]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months, y=revenue_trend,
            mode='lines+markers',
            name='Revenue (₹)',
            yaxis='y',
            line=dict(color='#00d4aa')
        ))
        
        fig.add_trace(go.Scatter(
            x=months, y=churn_trend,
            mode='lines+markers',
            name='Churn Rate (%)',
            yaxis='y2',
            line=dict(color='#ff6b6b')
        ))
        
        fig.update_layout(
            title="Monthly Trends",
            xaxis_title="Month",
            yaxis=dict(title="Revenue (₹)", side="left"),
            yaxis2=dict(title="Churn Rate (%)", side="right", overlaying="y"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Action Items
    st.markdown("## 🎯 Recommended Actions")
    
    actions = [
        {"priority": "High", "action": "Contact 156 at-risk customers for retention", "color": "#ff6b6b"},
        {"priority": "Medium", "action": "Launch premium service upgrade campaign", "color": "#feca57"},
        {"priority": "Low", "action": "Optimize network quality in East region", "color": "#00d4aa"}
    ]
    
    for action in actions:
        st.markdown(f"""
        <div class="info-card" style="border-left: 4px solid {action['color']};">
            <div style="display: flex; justify-content: between; align-items: center;">
                <div>
                    <strong style="color: {action['color']};">{action['priority']} Priority:</strong>
                    <p style="margin: 0.25rem 0 0 0;">{action['action']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Settings Page
def render_settings():
    st.markdown("# ⚙️ Settings & Preferences")
    
    st.markdown("""
    <div class="info-card">
        <h3>🎛️ Customize Your ForeTel.AI Experience</h3>
        <p>Configure your preferences, themes, and application settings.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme Settings
    st.markdown("## 🎨 Theme & Appearance")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🌙 Theme Selection")
        
        current_theme = st.session_state.theme
        theme_options = ["Dark", "Light", "Auto"]
        
        selected_theme = st.selectbox(
            "Choose Theme",
            theme_options,
            index=theme_options.index(current_theme.title()),
            help="Select your preferred color theme"
        )
        
        if selected_theme.lower() != current_theme:
            st.session_state.theme = selected_theme.lower()
            st.success(f"Theme changed to {selected_theme}")
            st.rerun()
        
        # Theme Preview
        if st.session_state.theme == "dark":
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #667eea;">
                <h4 style="color: #667eea;">Dark Theme Preview</h4>
                <p>Dark backgrounds with light text for reduced eye strain</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #ffffff; color: #212529; padding: 1rem; border-radius: 8px; border: 1px solid #dee2e6;">
                <h4 style="color: #667eea;">Light Theme Preview</h4>
                <p>Light backgrounds with dark text for better readability</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎭 UI Preferences")
        
        animations_enabled = st.checkbox(
            "Enable Animations",
            value=st.session_state.user_settings['chart_animations'],
            help="Toggle smooth animations and transitions"
        )
        
        notifications_enabled = st.checkbox(
            "Enable Notifications",
            value=st.session_state.user_settings['notifications'],
            help="Show system notifications and alerts"
        )
        
        auto_refresh = st.checkbox(
            "Auto Refresh Data",
            value=st.session_state.user_settings['auto_refresh'],
            help="Automatically refresh data every 5 minutes"
        )
        
        sound_effects = st.checkbox(
            "Sound Effects",
            value=st.session_state.user_settings['sound_effects'],
            help="Play sound effects for interactions"
        )
        
        # Update settings
        st.session_state.user_settings.update({
            'chart_animations': animations_enabled,
            'notifications': notifications_enabled,
            'auto_refresh': auto_refresh,
            'sound_effects': sound_effects
        })
    
    # Application Settings
    st.markdown("## 🔧 Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌍 Localization")
        
        language = st.selectbox(
            "Language",
            ["English", "Hindi", "Spanish", "French", "German"],
            index=["English", "Hindi", "Spanish", "French", "German"].index(st.session_state.user_settings['language'])
        )
        
        timezone = st.selectbox(
            "Timezone",
            ["Asia/Kolkata", "UTC", "America/New_York", "Europe/London", "Asia/Tokyo"],
            index=["Asia/Kolkata", "UTC", "America/New_York", "Europe/London", "Asia/Tokyo"].index(st.session_state.user_settings['timezone'])
        )
        
        st.session_state.user_settings.update({
            'language': language,
            'timezone': timezone
        })
    
    with col2:
        st.markdown("### 📊 Data & Analytics")
        
        data_refresh_interval = st.slider(
            "Data Refresh Interval (minutes)",
            1, 60, 5,
            help="How often to refresh dashboard data"
        )
        
        chart_theme = st.selectbox(
            "Chart Color Scheme",
            ["Default", "Viridis", "Plasma", "Inferno", "Magma"],
            help="Color scheme for charts and visualizations"
        )
        
        max_records = st.number_input(
            "Max Records to Display",
            100, 10000, 1000, step=100,
            help="Maximum number of records to show in tables"
        )
    
    # Performance Settings
    st.markdown("## ⚡ Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Optimization")
        
        cache_enabled = st.checkbox("Enable Data Caching", value=True)
        lazy_loading = st.checkbox("Lazy Load Charts", value=True)
        compression = st.checkbox("Enable Data Compression", value=False)
        
    with col2:
        st.markdown("### 📈 Analytics Tracking")
        
        usage_analytics = st.checkbox("Usage Analytics", value=True)
        error_reporting = st.checkbox("Error Reporting", value=True)
        performance_monitoring = st.checkbox("Performance Monitoring", value=False)
    
    # Security Settings
    st.markdown("## 🔒 Security & Privacy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛡️ Authentication")
        
        session_timeout = st.slider("Session Timeout (hours)", 1, 24, 8)
        two_factor_auth = st.checkbox("Two-Factor Authentication", value=False)
        
    with col2:
        st.markdown("### 🔐 Data Privacy")
        
        data_encryption = st.checkbox("Data Encryption", value=True, disabled=True)
        audit_logging = st.checkbox("Audit Logging", value=True)
    
    # Export/Import Settings
    st.markdown("## 📤 Backup & Restore")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Export Settings", use_container_width=True):
            settings_json = json.dumps(st.session_state.user_settings, indent=2)
            st.download_button(
                "Download Settings",
                settings_json,
                "foretel_settings.json",
                "application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("📁 Import Settings", type=['json'])
        if uploaded_file is not None:
            try:
                imported_settings = json.loads(uploaded_file.read())
                st.session_state.user_settings.update(imported_settings)
                st.success("Settings imported successfully!")
            except Exception as e:
                st.error(f"Error importing settings: {e}")
    
    with col3:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.session_state.user_settings = {
                'notifications': True,
                'auto_refresh': True,
                'chart_animations': True,
                'sound_effects': False,
                'language': 'English',
                'timezone': 'Asia/Kolkata'
            }
            st.success("Settings reset to defaults!")
            st.rerun()
    
    # System Information
    st.markdown("## ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>📱 Application Info</h4>
            <p><strong>Version:</strong> 2.0.0</p>
            <p><strong>Build:</strong> 2025.09.20</p>
            <p><strong>Environment:</strong> Production</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🔧 System Status</h4>
            <p><strong>Database:</strong> <span style="color: #00d4aa;">Connected</span></p>
            <p><strong>API:</strong> <span style="color: #00d4aa;">Online</span></p>
            <p><strong>Cache:</strong> <span style="color: #00d4aa;">Active</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Save Settings Button
    st.markdown("---")
    if st.button("💾 Save All Settings", use_container_width=True, type="primary"):
        st.success("✅ All settings saved successfully!")
        st.balloons()

# Apply theme-based CSS
def apply_theme_css():
    if st.session_state.theme == "light":
        light_theme_css = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        }
        
        .metric-card, .info-card, .feature-card {
            background: rgba(255, 255, 255, 0.9) !important;
            color: #2d3748 !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
        }
        
        .chat-message.bot {
            background: rgba(255, 255, 255, 0.9) !important;
            color: #2d3748 !important;
        }
        
        .main-header {
            background: rgba(255, 255, 255, 0.9) !important;
            color: #2d3748 !important;
        }
        </style>
        """
        st.markdown(light_theme_css, unsafe_allow_html=True)

# Main Application
def main():
    # Apply theme CSS
    apply_theme_css()
    
    render_navigation()
    
    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Churn Prediction":
        render_churn_prediction()
    elif st.session_state.current_page == "Revenue Forecasting":
        render_revenue_forecasting()
    elif st.session_state.current_page == "Analytics Dashboard":
        render_analytics_demo()
    elif st.session_state.current_page == "AI Chat Assistant":
        render_chat_demo()
    elif st.session_state.current_page == "Reports & Insights":
        render_reports()
    elif st.session_state.current_page == "Settings":
        render_settings()
    else:
        st.info(f"Page '{st.session_state.current_page}' is under development!")

if __name__ == "__main__":
    main()
