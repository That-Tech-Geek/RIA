import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
import streamlit as st
import plotly.graph_objects as go

# Define a function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Define a function to preprocess the data
def preprocess_data(stock_data):
    stock_data = stock_data.fillna(method='ffill')
    stock_data['MA_10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    return stock_data

# Define a function to build and train a linear regression model
def build_linear_regression_model(stock_data):
    X = stock_data[['MA_10', 'MA_50', 'MA_200', 'Daily_Return']].dropna()
    y = stock_data['Close'].loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

# Define a function to fetch news sentiment analysis using a mock API
def fetch_news_sentiment(ticker):
    # Mock sentiment analysis for demonstration purposes
    sentiments = ['Positive', 'Negative', 'Neutral']
    return np.random.choice(sentiments)

# Define a function to generate investment recommendations
def generate_recommendations(stock_data, model, ticker):
    sentiment = fetch_news_sentiment(ticker)
    latest_data = stock_data.iloc[-1]
    
    # Prepare the features for prediction
    latest_features = [[latest_data['MA_10'], latest_data['MA_50'], latest_data['MA_200'], latest_data['Daily_Return']]]
    
    # Check for NaN values
    if any(pd.isna(latest_features[0])):
        st.error("Latest data contains NaN values. Please check the stock data.")
        return "Unable to generate recommendation due to missing data."
    
    # Ensure the features are in the correct format
    latest_features = np.array(latest_features, dtype=float).reshape(1, -1)  # Ensure it's 2D
    
    # Make the prediction
    try:
        prediction = model.predict(latest_features)
    except ValueError as e:
        st.error(f"Prediction error: {e}")
        return "Unable to generate recommendation due to prediction error."
    
    if prediction > latest_data['Close'] and sentiment == 'Positive':
        return 'Buy'
    elif prediction < latest_data['Close'] and sentiment == 'Negative':
        return 'Sell'
    else:
        return 'Hold'
        
# Define a function to visualize the stock data and predictions using Plotly
def visualize_data(stock_data, model):
    X = stock_data[['MA_10', 'MA_50', 'MA_200', 'Daily_Return']].dropna()
    y = stock_data['Close'].loc[X.index]
    y_pred = model.predict(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=X.index, y=y_pred, mode='lines', name='Predicted', line=dict(color='orange')))
    
    fig.update_layout(title='Stock Price Prediction',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      legend=dict(x=0, y=1),
                      hovermode='x unified')
    
    st.plotly_chart(fig)
    
# Streamlit dashboard
def main():
    st.title("Investment Advisor Dashboard")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Start Date:", pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date:", pd.to_datetime('2025-02-05'))
    
    if st.button("Analyze"):
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        if stock_data.empty:
            st.error("No data found for the given ticker and date range.")
            return
        
        preprocessed_data = preprocess_data(stock_data)
        model, mse = build_linear_regression_model(preprocessed_data)
        recommendation = generate_recommendations(preprocessed_data, model, ticker)
        
        st.write(f'Mean Squared Error: {mse:.2f}')
        st.write(f'Investment Recommendation for {ticker}: {recommendation}')
        
        visualize_data(preprocessed_data, model)

if __name__ == '__main__':
    main()
