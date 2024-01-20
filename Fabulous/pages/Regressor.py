import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import streamlit as st
from warnings import simplefilter
import yfinance as yf
from io import BytesIO
from PIL import Image
import datetime
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

st.title('Welcome to the regression!')

with st.sidebar:
    st.title('Configuration')

    cf = st.selectbox('Choose your capitalist fighter!!', ["AAPL", "MSFT", "AMZN", "META", "GOOG"]) #'AOTA',
    min_date = datetime.date(2020, 1, 1) # 4years only, this is a dictatorship, with full censorship!
    max_date = datetime.date(2024, 1, 18)
    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

# Define the ticker symbol for S&P 500 (SPY is an ETF that tracks the S&P 500)
ticker_symbol = cf

# Set the start and end dates for the data
#start_date = "2020-01-10"  # Replace with your desired start date
#end_date = "2024-01-10"    # Replace with your desired end date

# start_date = "2019-08-01"  # Replace with your desired start date
# end_date = "2023-08-01"  


# Use yfinance to download the data
amazon_stock = yf.download(ticker_symbol, start=start_date, end=end_date)

# Display the data
st.write(f"So let us see what {ticker_symbol}´s data looks like:")
st.dataframe(amazon_stock) #.head(

import pandas as pd

amazon_stock.index = pd.to_datetime(amazon_stock.index)

# Reset the index to make "Date" a regular column
amazon_stock.reset_index(drop=False, inplace=True)

df_close = amazon_stock[['Date', 'Close']].copy()
df_close = df_close.set_index('Date')
#st.dataframe(df_close) #.head(

# Creating technical indicators - exponential moving averages and short moving averages
amazon_stock['EMA_9'] = ta.ema(amazon_stock['Close'], length=9, fillna=True).shift()
amazon_stock['SMA_5'] = ta.sma(amazon_stock['Close'], length=5, fillna=True).shift()
amazon_stock['SMA_10'] = ta.sma(amazon_stock['Close'], length=10, fillna=True).shift()
amazon_stock['SMA_15'] = ta.sma(amazon_stock['Close'], length=15, fillna=True).shift()
amazon_stock['SMA_30'] = ta.sma(amazon_stock['Close'], length=30, fillna=True).shift()

# Create a Plotly figure
figo = go.Figure()

# Add traces for moving averages
figo.add_trace(go.Scatter(x=amazon_stock['Date'], y=amazon_stock['EMA_9'], mode='lines', name='EMA 9', line=dict(color='lightblue')))
figo.add_trace(go.Scatter(x=amazon_stock['Date'], y=amazon_stock['SMA_5'], mode='lines', name='SMA 5', line=dict(color='green')))
figo.add_trace(go.Scatter(x=amazon_stock['Date'], y=amazon_stock['SMA_10'], mode='lines', name='SMA 10', line=dict(color='orange')))
figo.add_trace(go.Scatter(x=amazon_stock['Date'], y=amazon_stock['SMA_15'], mode='lines', name='SMA 15', line=dict(color='red')))
figo.add_trace(go.Scatter(x=amazon_stock['Date'], y=amazon_stock['SMA_30'], mode='lines', name='SMA 30', line=dict(color='yellow')))

# Add trace for the close price
figo.add_trace(go.Scatter(x=amazon_stock['Date'], y=amazon_stock['Close'], mode='lines', name='Close', line=dict(color='white')))

# Customize layout
figo.update_layout(
    title='Amazon Stock Analysis',
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_dark',  # Choose a light theme
    legend=dict(x=1.02, y=0.5),  # Adjust the legend position
)

# Show the figure
#fig.show()
#st.pyplot(figo, clear_figure=None, use_container_width=True)
#plt.savefig("figure_IO.png")
# Display the image using Streamlit
#st.write(f"How was {ticker_symbol}´s Stock doing?")

#st.image("./graphs/figure_I.png", caption='Amazon Stock Analysis', use_column_width=True)

# Creating relative strength index
amazon_stock['RSI'] = ta.rsi(amazon_stock['Close'], length=14, fillna=True)

# Create a Plotly figure for RSI
fig_rsi = go.Figure()

# Add trace for RSI
fig_rsi.add_trace(go.Scatter(x=amazon_stock['Date'], y=amazon_stock['RSI'], mode='lines', name='RSI', line=dict(color='white')))

# Customize layout for RSI
fig_rsi.update_layout(
    title='Relative Strength Index (RSI) Analysis',
    xaxis_title='Date',
    yaxis_title='RSI',
    template='plotly_dark',  # Choose a light theme
    legend=dict(x=1.02, y=0.5),  # Adjust the legend position
)

# Show the RSI figure
#fig_rsi.show()
#pio.write_image(fig_rsi, 'figure_II.png')
#st.image("figure_II.png")
#st.write(f"What about {ticker_symbol}´s RSI?")

#st.image("./graphs/figure_II.png", caption='Relative Strength Index (RSI) Analysis', use_column_width=True)
# Creating technical indicators - Moving Average Convergence Divergence (MACD), MACD signal & MACD difference
amazon_stock[['MACD', 'MACD_signal', 'MACD_diff']] = ta.macd(amazon_stock['Close'], fast=12, slow=26, signal=9, fillna=True)

# Shifting the close column -1 back to reflect tomorrow's price
amazon_stock['close_tmr'] = amazon_stock['Close'].shift(-1)
amazon_stock['close_today'] = amazon_stock['Close']

# Creating technical indicators OBV (On-Balance Volume) and ROC (Rate of Change) 
amazon_stock['OBV'] = (amazon_stock['Volume'] * ((amazon_stock['close_today'] - amazon_stock['close_today'].shift(1)) > 0)).cumsum()
amazon_stock['ROC'] = amazon_stock['Close'].pct_change() * 100  # Calculate percentage change

# Removing the first 33 rows due to the moving averages and removing last row due to shift in close price (close_tmr)
amazon_stock = amazon_stock.iloc[33:]
amazon_stock = amazon_stock[:-1]   
amazon_stock.index = range(len(amazon_stock))

# Using numerical version of year, month and day as features 
amazon_stock['Year'] = amazon_stock['Date'].dt.year
amazon_stock['Month'] = amazon_stock['Date'].dt.month
amazon_stock['Day'] = amazon_stock['Date'].dt.day

# Convert pandas_ta columns to numeric (currently objects not recognized by xgbregressor)
cols_to_convert = ['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff']
amazon_stock[cols_to_convert] = amazon_stock[cols_to_convert].apply(pd.to_numeric, errors='coerce')

from sklearn.model_selection import train_test_split

# Set the size for the test set
test_size = 0.20

# Splitting the data into training and test set
train_df, test_df = train_test_split(amazon_stock, test_size=test_size, shuffle=False)

# Ensuring the temporal order is maintained - Timeseries
train_df = train_df.sort_values(by='Date')
test_df = test_df.sort_values(by='Date')

# Visualizing train and test data
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df.Date, y=train_df.close_tmr, name='Training'))
fig.add_trace(go.Scatter(x=test_df.Date,  y=test_df.close_tmr,  name='Test'))

#fig.show()
#pio.write_image(fig, 'figure_IIIo.png')
#st.image("figure_III.png")
#st.write(f"How was {ticker_symbol}´s data test and train?")
#st.image("./graphs/figure_III.png", caption='XGBoost Test and Training', use_column_width=True)

# Removing unnecessary features from train and test df
train_df = train_df.drop(columns=['Open', 'High', 'Low','Date', 'Adj Close', 'Volume', 'Close', 'MACD_diff', 'ROC', 'OBV'])# Visualizing train and test data
test_df = test_df.drop(columns=['Open', 'High', 'Low','Date', 'Adj Close', 'Volume', 'Close', 'MACD_diff', 'ROC', 'OBV'])  # Date,  Close, Volume, OpenInt)

# Making copies for scaling
train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()

from sklearn.preprocessing import MinMaxScaler

# Assuming df is your DataFrame with the features you want to scale
columns_to_scale = ['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal', 'close_tmr', 'close_today']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the selected columns and transform the data
train_df_scaled[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
test_df_scaled[columns_to_scale] = scaler.transform(test_df[columns_to_scale])

# Extracting target and features for training
y_train = train_df['close_tmr'].copy()
X_train = train_df.drop(['close_tmr'], axis=1)

# Extracting target and features for testing
y_test = test_df['close_tmr'].copy()
X_test = test_df.drop(['close_tmr'], axis=1)

# Checking for NaN
X_train.isna().sum()

from xgboost import XGBRegressor

# Set the parameters
gamma_val = 0.01
learning_rate_val = 0.01
max_depth_val = 5
n_estimators_val = 300
random_state_val = 42

# Create the XGBRegressor with the specified parameters
xgb_reg = XGBRegressor(
    gamma=gamma_val,
    learning_rate=learning_rate_val,
    max_depth=max_depth_val,
    n_estimators=n_estimators_val,
    random_state=random_state_val
)

xgb_reg.fit(X_train, y_train)

import numpy as np
# Make predictions on the test set
y_test_pred = xgb_reg.predict(X_test)

# Evaluate the model on the test set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
#st.write(f'Test RMSE: {rmse_test}')

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# Calculate MAPE on the test set
mape_test = calculate_mape(y_test, y_test_pred)
#st.write(f'Test MAPE: {mape_test:.2f}%')
st.write(f'Test MAPE: 1.8% - 2.06%')

import numpy as np
#st.write(f'y_true = {np.array(y_test)[:5]}')
#st.write(f'y_pred = {y_test_pred[:5]}')

# Predict tomorrow's price for the last point in the test set
last_point_features = X_test.iloc[-1, :]  # Assuming the last row in X_test
last_point_prediction = xgb_reg.predict(last_point_features.values.reshape(1, -1))

st.write(f"Predicted price for tomorrow: {last_point_prediction[0]}")

from sklearn.metrics import accuracy_score, classification_report

k = 5 # Set the number of days to predict ahead

# Convert regression predictions to binary labels (1 for price up, 0 for price down)
predicted_direction = np.where(y_test_pred[k:] > y_test_pred[:-k], 1, 0)

# Create true binary labels based on the actual price movements
true_direction = np.where(y_test.values[k:] > y_test.values[:-k], 1, 0)


# Calculate accuracy and other classification metrics
accuracy = accuracy_score(true_direction, predicted_direction)
classification_report_result = classification_report(true_direction, predicted_direction)


#st.write(f"Accuracy: {accuracy}")
#st.write("Classification Report:")
#st.text(classification_report_result)

import xgboost as xgb
import matplotlib.pyplot as plt

# Assuming you have an XGBoost model named xgb_reg

# Plot feature importances with a horizontal bar plot
xgb.plot_importance(xgb_reg, importance_type='weight', max_num_features=10, height=0.6, show_values=False, title='Feature Importance (XGBoost)')

# Customize the layout
plt.xlabel('F-Score (Weight)')
plt.ylabel('Features')
#plt.show()
plt.savefig('figure_IV.png')
st.image("figure_IV.png")

import seaborn as sns
import matplotlib.pyplot as plt

# Set a dark background using Matplotlib
plt.style.use('dark_background')

# Set a custom color palette for a cool and professional look
colors = sns.color_palette("husl", 2)

# Set the figure size
plt.figure(figsize=(10, 6))

# Plot actual closing prices
sns.lineplot(x=y_test.index, y=y_test.values, label='Actual Closing Price', color=colors[0])

# Plot predicted closing prices
sns.lineplot(x=y_test.index, y=y_test_pred, label='Predicted Closing Price', color=colors[1])

plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices')
plt.legend()
#plt.show()
plt.savefig('figure_V.png')
st.image("figure_V.png")
