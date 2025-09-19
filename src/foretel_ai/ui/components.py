"""
Professional UI components for ForeTel.AI Streamlit application.

This module provides reusable UI components with consistent styling,
interactive elements, and professional data visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Union, Tuple
import base64
from io import BytesIO

from ..config.settings import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class UIComponents:
    """Professional UI components for ForeTel.AI application."""
    
    @staticmethod
    def page_header(title: str, subtitle: str = "", icon: str = "📊") -> None:
        """
        Create a professional page header.
        
        Args:
            title: Main page title
            subtitle: Optional subtitle
            icon: Page icon
        """
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, {config.ui.primary_color} 0%, {config.ui.secondary_color} 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h1 style="
                color: white;
                margin: 0;
                font-size: 2.5rem;
                font-weight: 700;
            ">{icon} {title}</h1>
            {f'<p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.2rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metric_card(
        title: str, 
        value: Union[str, int, float], 
        delta: Optional[str] = None,
        delta_color: str = "normal",
        icon: str = "📈"
    ) -> None:
        """
        Create a professional metric card.
        
        Args:
            title: Metric title
            value: Metric value
            delta: Change indicator
            delta_color: Delta color (normal, inverse, off)
            icon: Metric icon
        """
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(f"""
            <div style="
                font-size: 3rem;
                text-align: center;
                padding: 1rem;
            ">{icon}</div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color
            )
    
    @staticmethod
    def info_box(message: str, box_type: str = "info") -> None:
        """
        Create an information box.
        
        Args:
            message: Message to display
            box_type: Type of box (info, success, warning, error)
        """
        colors = {
            "info": ("#e3f2fd", "#1976d2"),
            "success": ("#e8f5e8", "#2e7d32"),
            "warning": ("#fff3e0", "#f57c00"),
            "error": ("#ffebee", "#d32f2f")
        }
        
        bg_color, text_color = colors.get(box_type, colors["info"])
        
        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            color: {text_color};
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid {text_color};
            margin: 1rem 0;
        ">
            {message}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def data_table(
        df: pd.DataFrame,
        title: str = "Data Table",
        searchable: bool = True,
        downloadable: bool = True,
        max_rows: int = 1000
    ) -> None:
        """
        Create a professional data table with search and download functionality.
        
        Args:
            df: DataFrame to display
            title: Table title
            searchable: Enable search functionality
            downloadable: Enable download functionality
            max_rows: Maximum rows to display
        """
        st.subheader(title)
        
        # Search functionality
        if searchable and len(df) > 0:
            search_term = st.text_input("🔍 Search data", key=f"search_{title}")
            if search_term:
                # Search across all string columns
                string_cols = df.select_dtypes(include=['object']).columns
                mask = df[string_cols].astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                df = df[mask]
        
        # Display table
        if len(df) > max_rows:
            st.warning(f"Showing first {max_rows} rows of {len(df)} total rows")
            df_display = df.head(max_rows)
        else:
            df_display = df
        
        st.dataframe(
            df_display,
            use_container_width=True,
            height=min(400, len(df_display) * 35 + 100)
        )
        
        # Download functionality
        if downloadable and len(df) > 0:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{title.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    @staticmethod
    def create_gauge_chart(
        value: float,
        title: str,
        min_val: float = 0,
        max_val: float = 100,
        thresholds: List[float] = [30, 70],
        colors: List[str] = ["red", "yellow", "green"]
    ) -> go.Figure:
        """
        Create a gauge chart for displaying metrics.
        
        Args:
            value: Current value
            title: Chart title
            min_val: Minimum value
            max_val: Maximum value
            thresholds: Threshold values for color zones
            colors: Colors for each zone
            
        Returns:
            Plotly gauge chart figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': (min_val + max_val) / 2},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, thresholds[0]], 'color': colors[0]},
                    {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                    {'range': [thresholds[1], max_val], 'color': colors[2]}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        return fig
    
    @staticmethod
    def create_distribution_chart(
        data: pd.Series,
        title: str,
        chart_type: str = "histogram"
    ) -> go.Figure:
        """
        Create distribution charts (histogram, box plot, violin plot).
        
        Args:
            data: Data series
            title: Chart title
            chart_type: Type of chart (histogram, box, violin)
            
        Returns:
            Plotly figure
        """
        if chart_type == "histogram":
            fig = px.histogram(
                x=data,
                title=title,
                nbins=30,
                color_discrete_sequence=[config.ui.primary_color]
            )
        elif chart_type == "box":
            fig = px.box(
                y=data,
                title=title,
                color_discrete_sequence=[config.ui.primary_color]
            )
        elif chart_type == "violin":
            fig = px.violin(
                y=data,
                title=title,
                color_discrete_sequence=[config.ui.primary_color]
            )
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        fig.update_layout(
            height=config.ui.chart_height,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            df: DataFrame with numerical columns
            title: Chart title
            
        Returns:
            Plotly heatmap figure
        """
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            st.warning("No numerical columns found for correlation analysis")
            return go.Figure()
        
        corr_matrix = numerical_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            title=title,
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        fig.update_layout(
            height=config.ui.chart_height,
            width=config.ui.chart_width
        )
        
        return fig
    
    @staticmethod
    def create_time_series_chart(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        color_col: Optional[str] = None
    ) -> go.Figure:
        """
        Create a time series chart.
        
        Args:
            df: DataFrame with time series data
            x_col: X-axis column (time)
            y_col: Y-axis column (values)
            title: Chart title
            color_col: Optional color grouping column
            
        Returns:
            Plotly line chart figure
        """
        if color_col:
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title
            )
        else:
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=[config.ui.primary_color]
            )
        
        fig.update_layout(
            height=config.ui.chart_height,
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_chart(
        importance_dict: Dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 15
    ) -> go.Figure:
        """
        Create a feature importance chart.
        
        Args:
            importance_dict: Dictionary of feature names and importance scores
            title: Chart title
            top_n: Number of top features to display
            
        Returns:
            Plotly bar chart figure
        """
        # Sort and select top features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker_color=config.ui.primary_color
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(features) * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    @staticmethod
    def prediction_result_card(
        prediction: Dict[str, Any],
        card_type: str = "churn"
    ) -> None:
        """
        Create a prediction result card.
        
        Args:
            prediction: Prediction result dictionary
            card_type: Type of prediction (churn, revenue)
        """
        if card_type == "churn":
            risk_color = {
                "Very Low": "#4CAF50",
                "Low": "#8BC34A", 
                "Medium": "#FF9800",
                "High": "#FF5722",
                "Very High": "#F44336"
            }.get(prediction.get('risk_level', 'Medium'), "#FF9800")
            
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 5px solid {risk_color};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 1rem 0;
            ">
                <h3 style="margin: 0 0 1rem 0; color: {config.ui.text_color};">
                    🎯 Churn Prediction Result
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <strong>Churn Probability:</strong><br>
                        <span style="font-size: 1.5rem; color: {risk_color};">
                            {prediction.get('churn_probability', 0):.1%}
                        </span>
                    </div>
                    <div>
                        <strong>Risk Level:</strong><br>
                        <span style="font-size: 1.2rem; color: {risk_color};">
                            {prediction.get('risk_level', 'Unknown')}
                        </span>
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <strong>Confidence:</strong> {prediction.get('confidence', 0):.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif card_type == "revenue":
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 5px solid {config.ui.secondary_color};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 1rem 0;
            ">
                <h3 style="margin: 0 0 1rem 0; color: {config.ui.text_color};">
                    💰 Revenue Forecast Result
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <strong>Predicted Revenue:</strong><br>
                        <span style="font-size: 1.5rem; color: {config.ui.secondary_color};">
                            ${prediction.get('predicted_revenue', 0):,.2f}
                        </span>
                    </div>
                    <div>
                        <strong>Confidence:</strong><br>
                        <span style="font-size: 1.2rem;">
                            {prediction.get('model_confidence', 0):.1%}
                        </span>
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <strong>Forecast Period:</strong> {prediction.get('forecast_period', 'Monthly')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def sidebar_navigation() -> str:
        """
        Create sidebar navigation menu.
        
        Returns:
            Selected page name
        """
        st.sidebar.markdown(f"""
        <div style="
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, {config.ui.primary_color}, {config.ui.secondary_color});
            border-radius: 10px;
            margin-bottom: 2rem;
        ">
            <h2 style="color: white; margin: 0;">ForeTel.AI</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                Telecom Analytics Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        pages = {
            "🏠 Home": "home",
            "📊 Churn Prediction": "churn",
            "💰 Revenue Forecasting": "revenue", 
            "📈 Analytics Dashboard": "analytics",
            "🤖 AI Assistant": "chat",
            "📄 Document Insights": "documents"
        }
        
        selected = st.sidebar.selectbox(
            "Navigate to:",
            list(pages.keys()),
            key="navigation"
        )
        
        return pages[selected]
    
    @staticmethod
    def loading_spinner(message: str = "Processing...") -> None:
        """
        Display a loading spinner with message.
        
        Args:
            message: Loading message
        """
        with st.spinner(message):
            st.empty()
    
    @staticmethod
    def progress_bar(progress: float, message: str = "") -> None:
        """
        Display a progress bar.
        
        Args:
            progress: Progress value (0.0 to 1.0)
            message: Progress message
        """
        if message:
            st.text(message)
        st.progress(progress)

class DataVisualization:
    """Advanced data visualization components."""
    
    @staticmethod
    def customer_segmentation_chart(df: pd.DataFrame) -> go.Figure:
        """Create customer segmentation visualization."""
        if 'MonthlySpending' not in df.columns or 'TenureMonths' not in df.columns:
            return go.Figure()
        
        fig = px.scatter(
            df,
            x='TenureMonths',
            y='MonthlySpending',
            color='Churned' if 'Churned' in df.columns else None,
            size='DataUsageGB' if 'DataUsageGB' in df.columns else None,
            title="Customer Segmentation Analysis",
            labels={
                'TenureMonths': 'Tenure (Months)',
                'MonthlySpending': 'Monthly Spending ($)',
                'Churned': 'Churn Status'
            }
        )
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def revenue_trend_chart(df: pd.DataFrame) -> go.Figure:
        """Create revenue trend visualization."""
        if 'TenureMonths' not in df.columns or 'TotalRevenue' not in df.columns:
            return go.Figure()
        
        # Aggregate revenue by tenure
        revenue_trend = df.groupby('TenureMonths')['TotalRevenue'].agg(['mean', 'sum']).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Average Revenue per Customer', 'Total Revenue'],
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=revenue_trend['TenureMonths'],
                y=revenue_trend['mean'],
                mode='lines+markers',
                name='Avg Revenue',
                line=dict(color=config.ui.primary_color)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=revenue_trend['TenureMonths'],
                y=revenue_trend['sum'],
                mode='lines+markers',
                name='Total Revenue',
                line=dict(color=config.ui.secondary_color)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title_text="Revenue Trends by Customer Tenure"
        )
        
        return fig

__all__ = ['UIComponents', 'DataVisualization']
