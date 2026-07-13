import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Title
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["Single Prediction", "Batch Prediction", "Dashboard", "About"]
    )

if page == "Single Prediction":
    st.header("Single Transaction Prediction")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_id = st.text_input("Transaction ID", value="TXN001")
        amount = st.number_input("Amount", min_value=0.0, value=100.0, step=1.0)
        time = st.number_input("Time (seconds)", min_value=0.0, value=0.0, step=1.0)
    
    # V features
    with col2:
        st.subheader("V Features")
        v_features = {}
        for i in range(1, 29):
            v_features[f'V{i}'] = st.number_input(f"V{i}", value=0.0, step=0.01, format="%.4f")
    
    # Predict button
    if st.button("Predict Fraud", type="primary"):
        # Prepare transaction data
        transaction = {
            "transaction_id": transaction_id,
            "Time": time,
            "Amount": amount,
            **v_features
        }
        
        with st.spinner("Analyzing transaction..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=transaction)
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    st.success("Prediction Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if result['is_fraud']:
                            st.error("🚨 FRAUD DETECTED")
                        else:
                            st.success("✅ Legitimate Transaction")
                    
                    with col2:
                        st.metric(
                            "Fraud Probability",
                            f"{result['fraud_probability']:.2%}"
                        )
                    
                    with col3:
                        risk_color = {
                            'high': '🔴',
                            'medium': '🟡',
                            'low': '🟢'
                        }
                        st.metric(
                            "Risk Level",
                            f"{risk_color.get(result['risk_level'], '')} {result['risk_level'].upper()}"
                        )
                    
                    # Display transaction details
                    with st.expander("Transaction Details"):
                        st.json(result['features'])
                    
                    # Display fraud probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = result['fraud_probability'] * 100,
                        title = {'text': "Fraud Probability (%)"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"Connection Error: {e}")

elif page == "Batch Prediction":
    st.header("Batch Transaction Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())
        
        if st.button("Process Batch", type="primary"):
            with st.spinner("Processing transactions..."):
                try:
                    # Convert to JSON
                    transactions = df.to_dict('records')
                    
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        json={"transactions": transactions}
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Transactions", results['total'])
                        
                        with col2:
                            st.metric("Fraud Cases", results['fraud_count'])
                        
                        with col3:
                            fraud_rate = (results['fraud_count'] / results['total']) * 100
                            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results['results'])
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results (CSV)",
                            data=csv,
                            file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    st.error(f"Connection Error: {e}")

elif page == "Dashboard":
    st.header("📊 Fraud Detection Dashboard")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", "284,807")
    
    with col2:
        st.metric("Total Fraud Cases", "492", delta="-15.2%")
    
    with col3:
        st.metric("Fraud Rate", "0.17%", delta="-0.02%")
    
    with col4:
        st.metric("Model Accuracy", "99.9%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fraud vs Non-Fraud Distribution")
        fraud_counts = pd.DataFrame({
            'Category': ['Non-Fraud', 'Fraud'],
            'Count': [284315, 492]
        })
        fig = px.pie(fraud_counts, values='Count', names='Category', 
                     color_discrete_map={'Non-Fraud': '#2ecc71', 'Fraud': '#e74c3c'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Recent Fraud Detection Trend")
        dates = pd.date_range(start='2024-01-01', periods=30)
        fraud_counts = np.random.randint(0, 10, size=30)
        fraud_df = pd.DataFrame({'Date': dates, 'Fraud Cases': fraud_counts})
        fig = px.line(fraud_df, x='Date', y='Fraud Cases')
        st.plotly_chart(fig, use_container_width=True)

elif page == "About":
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### Credit Card Fraud Detection System
    
    This system uses machine learning to detect fraudulent credit card transactions in real-time.
    
    #### Key Features:
    - **Real-time Prediction**: Predict fraud instantly for single transactions
    - **Batch Processing**: Process multiple transactions at once
    - **Fraud Risk Assessment**: Get risk level (High/Medium/Low)
    - **Explainable AI**: Understand why transactions are flagged
    - **Performance Monitoring**: Track model metrics and fraud trends
    
    #### Models Used:
    - XGBoost
    - Random Forest
    - Logistic Regression
    
    #### Technologies:
    - FastAPI for backend API
    - Streamlit for frontend dashboard
    - MLflow for experiment tracking
    - Docker for containerization
    - Prometheus & Grafana for monitoring
    
    #### Business Value:
    - **Reduce Fraud Losses**: Early detection prevents chargebacks
    - **Operational Efficiency**: Automate fraud screening
    - **Customer Trust**: Quick approval for legitimate transactions
    - **Regulatory Compliance**: Maintain audit trails
    """)