import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, r2_score
import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from scikeras.wrappers import KerasRegressor
# Load environment variables from the .env file
load_dotenv('../.env')

# Data Viz. 
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.ndimage import gaussian_filter
from calendar import monthrange
from calendar import month_name

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import requests
import csv
from itertools import permutations
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf

import statsmodels.graphics.tsaplots as tsaplot
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing, SimpleExpSmoothing

import warnings
warnings.filterwarnings('ignore')
  

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_ta import *
from sklearn.linear_model import LinearRegression

def lstm_predict(cf, start_date, end_date, k, color):
    # LSTM magic here
    # Fetch historical stock data
    import yfinance as yf
    df = yf.download(cf, start=start_date , end=end_date)

    # Calculate daily returns
    df['returns'] = df['Adj Close'].pct_change()

    # Fetch market data (e.g., S&P 500)
    market_data = yf.download('^GSPC', start=df.index.min(), end=df.index.max())
    market_data['market_returns'] = market_data['Adj Close'].pct_change()

    # Combine stock and market data
    merged_data = pd.merge(df, market_data[['market_returns']], left_index=True, right_index=True, how='inner')

    # Drop rows with missing values
    merged_data.dropna(inplace=True)

    # Initialize lists to store alpha and beta values
    alpha_values = []
    beta_values = []

    # Set up X and y for linear regression
    X = merged_data['market_returns'].values.reshape(-1, 1)
    y = merged_data['returns'].values

    # Iterate through the data to calculate alpha and beta for each day
    for i in range(len(merged_data)):
        X_i = X[:i + 1]
        y_i = y[:i + 1]

        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_i, y_i)

        # Extract alpha and beta for the current day
        alpha_i = model.intercept_
        beta_i = model.coef_[0]

        alpha_values.append(alpha_i)
        beta_values.append(beta_i)

    # Add alpha and beta columns to the DataFrame
    merged_data['alpha'] = alpha_values
    merged_data['beta'] = beta_values

    # Drop columns not needed for the final result
    merged_data.drop(['returns', 'market_returns'], axis=1, inplace=True)

    # Add technical analysis features
    import pandas_ta as ta
    import pandas
    from ta import add_all_ta_features
    merged_data = add_all_ta_features(merged_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    # Calculate target and target class
    merged_data['target'] = ((merged_data['Close'] - merged_data['Open']) / merged_data['Open']) * 100
    merged_data['target'] = merged_data['target'].shift(-1)
 
    merged_data['target_class'] = np.where(merged_data['target'] < 0, 0, 1)
    
    merged_data['target_next_close'] = merged_data['Close'].shift(-1)

    # Drop rows with missing values
    merged_data.dropna(inplace=True)

    #return merged_data
    import pandas_ta
    from pandas_ta import add_all_ta_features
    stock_data = load_and_prepare('AMZN', "2010-01-01" , "2024-01-01")
    st.dataframe(stock_data)
    

    df = stock_data
    

    df.isna().sum()


    from sklearn.model_selection import train_test_split

    # Assuming X is your feature set and y is your target variable
    X = df.drop(['Open', 'High', 'Low', 'Close', 'Volume','target',
        'target_class', 'target_next_close'], axis=1)# Adjust columns accordingly

    y = df[['target_class']]  # Adjust the target variable accordingly

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=42)

    from sklearn.model_selection import train_test_split

    # Extract features and target variable
    features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Adjust columns accordingly
    target = stock_data['target'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

    # Reshape the data for LSTM input (samples, time steps, features)
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=1)

    # Make predictions on the test set
    test_predictions_scaled = model.predict(X_test_lstm)

    # Inverse transform the predictions and actual values for comparison
    test_predictions = scaler.inverse_transform(test_predictions_scaled)
    y_test_inv = scaler.inverse_transform(y_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_inv, test_predictions)
    print(f'Mean Squared Error (MSE): {mse}')

    # Plot actual vs predicted values for the test set
    plt.plot(y_test_inv, label='Actual')
    plt.plot(test_predictions, label='Predicted')
    plt.title('Actual vs Predicted Percentage Change')
    plt.xlabel('Time Steps')
    plt.ylabel('Percentage Change')
    plt.legend()
    plt.show()
    # Check the scale of the features
    print("Feature Scale (min, max):", scaler.data_min_, scaler.data_max_)

    # Check the scale of the target variable
    print("Target Scale (min, max):", scaler.data_min_[0], scaler.data_max_[0])

    st.write(f"LSTM predictions for {cf} from {start_date} to {end_date} with {k} days into the future.") 
st.balloons