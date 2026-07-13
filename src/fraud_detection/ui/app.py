import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide"
)

# Title
st.title("🕵️ Credit Card Fraud Detection Dashboard")
st.markdown("Real-time monitoring and prediction of fraudulent transactions")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # API endpoint
    api_url = st.text_input("API URL", "http://localhost:8000")
    
    # Prediction threshold
    threshold = st.slider(
        "Fraud Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Probability threshold for classifying as fraud"
    )
    
    # Refresh interval
    refresh_interval = st.selectbox(
        "Refresh Interval",
        ["Realtime", "10 seconds", "30 seconds", "1 minute"],
        index=2
    )

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 Predict",
    "📈 Analytics",
    "⚙️ Model Info"
])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", "284,807", "100%")
    with col2:
        st.metric("Fraudulent", "492", "0.17%")
    with col3:
        st.metric("Detection Rate", "92.3%", "↑ 1.2%")
    with col4:
        st.metric("False Positives", "3.2%", "↓ 0.5%")
    
    # Time series chart
    st.subheader("Transaction Volume Over Time")
    
    # Generate sample data
    dates = pd.date_range(start="2023-01-01", periods=100, freq='H')
    volumes = np.random.poisson(50, 100)
    fraud_mask = np.random.rand(100) < 0.01
    
    df_time = pd.DataFrame({
        'timestamp': dates,
        'volume': volumes,
        'is_fraud': fraud_mask
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_time['timestamp'],
        y=df_time['volume'],
        mode='lines',
        name='All Transactions',
        line=dict(color='blue', width=2)
    ))
    
    # Highlight fraud points
    fraud_df = df_time[df_time['is_fraud']]
    fig.add_trace(go.Scatter(
        x=fraud_df['timestamp'],
        y=fraud_df['volume'],
        mode='markers',
        name='Fraud Detected',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="Transaction Count",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Transaction Fraud Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction ID
        transaction_id = st.text_input("Transaction ID", "txn_001")
        
        # Amount
        amount = st.number_input("Amount ($)", min_value=0.0, max_value=100000.0, value=149.62)
        
        # Time
        time = st.number_input("Time (seconds from first)", min_value=0.0, max_value=200000.0, value=0.0)
    
    with col2:
        # Feature inputs (simplified - in reality would have all V1-V28)
        st.markdown("**Transaction Features**")
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            v1 = st.number_input("V1", value=-1.359807)
            v2 = st.number_input("V2", value=-0.072781)
            v3 = st.number_input("V3", value=2.536347)
        
        with col2_2:
            v4 = st.number_input("V4", value=1.378155)
            v5 = st.number_input("V5", value=-0.338321)
            v6 = st.number_input("V6", value=0.462388)
    
    # Predict button
    if st.button("Predict Fraud", type="primary"):
        # Prepare transaction data
        transaction = {
            "transaction_id": transaction_id,
            "time": time,
            "v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5, "v6": v6,
            "v7": 0.239599, "v8": 0.098698, "v9": 0.363787,
            "v10": -0.090794, "v11": -0.551600, "v12": -0.617801,
            "v13": -0.991390, "v14": -0.311169, "v15": 1.468177,
            "v16": -0.470401, "v17": 0.207971, "v18": 0.025791,
            "v19": 0.403993, "v20": 0.251412, "v21": -0.018307,
            "v22": 0.277838, "v23": -0.110474, "v24": 0.066928,
            "v25": 0.128539, "v26": -0.189115, "v27": 0.133558,
            "v28": -0.021053, "amount": amount
        }
        
        try:
            # Call API
            response = requests.post(
                f"{api_url}/predict",
                json=transaction
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display result
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    # Probability gauge
                    prob = result['fraud_probability']
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Fraud Probability (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': threshold * 100
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col_result2:
                    # Result card
                    if result['is_fraud']:
                        st.error("🚨 **FRAUD DETECTED** 🚨")
                        st.markdown(f"**Probability**: {prob:.1%}")
                        st.markdown(f"**Threshold**: {threshold:.1%}")
                        st.markdown(f"**Transaction ID**: {result['transaction_id']}")
                        
                        # Alert actions
                        st.markdown("### Recommended Actions:")
                        st.markdown("1. 🛑 Block transaction immediately")
                        st.markdown("2. 📞 Contact cardholder")
                        st.markdown("3. 🏷️ Flag account for review")
                    else:
                        st.success("✅ **LEGITIMATE TRANSACTION** ✅")
                        st.markdown(f"**Probability**: {prob:.1%}")
                        st.markdown(f"**Threshold**: {threshold:.1%}")
                        st.markdown(f"**Transaction ID**: {result['transaction_id']}")
                        
                        # Explanation
                        if 'explanation' in result:
                            st.markdown("### Key Factors:")
                            exp = result['explanation']
                            if 'top_features' in exp:
                                for feature, importance in exp['top_features'].items():
                                    st.markdown(f"- **{feature}**: {importance:.3f}")
                
                # Show raw response
                with st.expander("View Raw Response"):
                    st.json(result)
            
            else:
                st.error(f"API Error: {response.status_code}")
                st.text(response.text)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab3:
    st.subheader("Model Analytics")
    
    # Model performance metrics
    metrics_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM'],
        'ROC AUC': [0.971, 0.958, 0.967, 0.975],
        'Recall': [0.918, 0.745, 0.857, 0.892],
        'Precision': [0.059, 0.961, 0.894, 0.912],
        'Business Cost': [1250, 980, 810, 750]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['ROC AUC', 'Recall'])
                          .highlight_min(axis=0, subset=['Business Cost']), 
                 use_container_width=True)
    
    # Cost analysis
    st.subheader("Business Cost Analysis")
    
    cost_fig = go.Figure(data=[
        go.Bar(name='False Negative Cost', x=df_metrics['Model'], 
               y=[100 * (1-r) * 98 for r in df_metrics['Recall']]),
        go.Bar(name='False Positive Cost', x=df_metrics['Model'], 
               y=[10 * (1-p) * 56864 for p in df_metrics['Precision']])
    ])
    
    cost_fig.update_layout(
        barmode='stack',
        title="Cost Breakdown by Model",
        yaxis_title="Total Cost ($)",
        height=400
    )
    
    st.plotly_chart(cost_fig, use_container_width=True)

with tab4:
    st.subheader("Model Information")
    
    try:
        # Get model info from API
        response = requests.get(f"{api_url}/model_info")
        
        if response.status_code == 200:
            model_info = response.json()
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown("### Model Details")
                st.markdown(f"**Name**: {model_info.get('model_name', 'N/A')}")
                st.markdown(f"**Training Date**: {model_info.get('training_date', 'N/A')}")
                st.markdown(f"**Version**: {model_info.get('version', '1.0.0')}")
            
            with col_info2:
                st.markdown("### Performance Metrics")
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    st.markdown(f"**ROC AUC**: {metrics.get('roc_auc', 'N/A'):.3f}")
                    st.markdown(f"**Recall**: {metrics.get('recall', 'N/A'):.3f}")
                    st.markdown(f"**Precision**: {metrics.get('precision', 'N/A'):.3f}")
            
            # Feature importance
            if 'features' in model_info:
                st.markdown("### Top 10 Features")
                features = model_info.get('feature_importance', {})
                
                if features:
                    df_features = pd.DataFrame({
                        'Feature': list(features.keys()),
                        'Importance': list(features.values())
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig_features = px.bar(
                        df_features,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance"
                    )
                    st.plotly_chart(fig_features, use_container_width=True)
        
        else:
            st.warning("Could not fetch model info from API")
    
    except:
        st.warning("API not available. Using sample data.")
        
        # Sample model info
        st.markdown("### Sample Model Information")
        st.json({
            "model_name": "XGBoost_Fraud_Detector",
            "training_date": "2023-12-01",
            "metrics": {
                "roc_auc": 0.975,
                "recall": 0.892,
                "precision": 0.912
            }
        })

# Footer
st.markdown("---")
st.markdown("### 📊 Live Monitoring")
st.markdown("""
This dashboard provides real-time insights into transaction patterns and fraud detection performance.
Adjust the threshold in the sidebar to balance between fraud detection and false positives.
""")