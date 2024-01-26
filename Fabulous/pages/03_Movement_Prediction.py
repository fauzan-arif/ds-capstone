import streamlit as st
import datetime
import os
import yfinance as yf
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

#from xgboost_page import xgboost_predict  
#from lstm_page import lstm_predict  
#from XGBClassifier_AAPL import xgboost_AAPL_predict
import pickle
from helper import save_model, load_model
# Greetings
st.title('Will the stock go up or down?')
#st.image('./pics/grievous.jpeg')
st.write('Configurations to be adjusted in the sidebar.')
# Sidebar enchantment
#st.snow()

with st.sidebar:
    st.title('Configuration')
    cf = st.selectbox('Choose your capitalist fighter!!', ["AAPL", "AMZN", "MSFT",  "META", "GOOG"]) 
    #period = st.selectbox("How far you wanna go back?", ['max',  '1d', '5d', '1mo'])
    max_date = datetime.date.today()
    min_date = max_date - datetime.timedelta(days=365*4)
    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

    #k = st.selectbox('How far back to the future you wanna go Doc?', [3, 5, 7, 10])
    #color = st.color_picker('Pick a color. Don’t look so sus, trust me yalla!')  # you really should not.
period = 'max'
# Continue the adventure
#st.text("Interesting choices. Shall we?")
#pickle_rick = load_model(f'../model_AAPL.pickle')
#prediction = pickle_rick.predict(input)
#st.button('Let's go back to the future Doc!')

from ta import add_all_ta_features
def load_data_from_yfinance(cf, period):
    data = yf.Ticker(cf).history(period=period)
    return data

def classify_prediction(prediction, threshold=0.52):
  return (prediction[:, 1] > threshold).astype(int)[0]

def image_for_classification(classification):
    return {
        0: "./pages/down.png",
        1: "./pages/up.png",
    }.get(classification) 

def get_model_for_prediction(cf):
    file = os.path.join("pages", "models", f"grid_model_{cf}.pickle")
    return load_model(file)


data = load_data_from_yfinance(cf, period)


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
#st.write(f"So let us see what {ticker_symbol}´s data looks like:")
#st.dataframe(amazon_stock) #.head(

import pandas as pd

amazon_stock.index = pd.to_datetime(amazon_stock.index)

# Reset the index to make "Date" a regular column
amazon_stock.reset_index(drop=False, inplace=True)

df_close = amazon_stock[['Date', 'Close']].copy()
df_close = df_close.set_index('Date')
#st.dataframe(df_close) #.head(

# Creating technical indicators - exponential moving averages and short moving averages
amazon_stock['RSI']=ta.rsi(amazon_stock.Close, length=15)
amazon_stock['EMAF']=ta.ema(amazon_stock.Close, length=20)
amazon_stock['EMAM']=ta.ema(amazon_stock.Close, length=100)
amazon_stock['EMAS']=ta.ema(amazon_stock.Close, length=150)
EMA_12 = pd.Series(amazon_stock['Close'].ewm(span=12, min_periods=12).mean())
EMA_26 = pd.Series(amazon_stock['Close'].ewm(span=26, min_periods=26).mean())
amazon_stock['MACD'] = pd.Series(EMA_12 - EMA_26)
amazon_stock['MACD_signal'] = pd.Series(amazon_stock.MACD.ewm(span=9, min_periods=9).mean())

# Shifting the close column -1 back to reflect tomorrow's price
amazon_stock['close_tmr'] = amazon_stock['Close'].shift(-1)
amazon_stock['close_today'] = amazon_stock['Close']

# Removing the first 33 rows due to the moving averages and removing last row due to shift in close price (close_tmr)
amazon_stock = amazon_stock.iloc[149:]
amazon_stock = amazon_stock[:-1]   
amazon_stock.index = range(len(amazon_stock))

# Using numerical version of year, month and day as features 
amazon_stock['Year'] = amazon_stock['Date'].dt.year
amazon_stock['Month'] = amazon_stock['Date'].dt.month
amazon_stock['Day'] = amazon_stock['Date'].dt.day

# Convert pandas_ta columns to numeric (currently objects not recognized by xgbregressor)
cols_to_convert = ['EMAF', 'EMAM', 'EMAS']
amazon_stock[cols_to_convert] = amazon_stock[cols_to_convert].apply(pd.to_numeric, errors='coerce')

from sklearn.model_selection import train_test_split

# Set the size for the test set
test_size = 0.20

# Splitting the data into training and test set
train_df, test_df = train_test_split(amazon_stock, test_size=test_size, shuffle=False)

# Ensuring the temporal order is maintained - Timeseries
train_df = train_df.sort_values(by='Date')
test_df = test_df.sort_values(by='Date')

# Removing unnecessary features from train and test df
train_df = train_df.drop(columns=['Open', 'High', 'Low','Date', 'Adj Close', 'Volume', 'Close'])# Visualizing train and test data
test_df = test_df.drop(columns=['Open', 'High', 'Low','Date', 'Adj Close', 'Volume', 'Close'])  # Date,  Close, Volume, OpenInt)

# Making copies for scaling
train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()

from sklearn.preprocessing import MinMaxScaler

# Extracting target and features for training
y_train = train_df['close_tmr'].copy()
X_train = train_df.drop(['close_tmr'], axis=1)

# Extracting target and features for testing
y_test = test_df['close_tmr'].copy()
X_test = test_df.drop(['close_tmr'], axis=1)

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


import numpy as np
#st.write(f'y_true = {np.array(y_test)[:5]}')
#st.write(f'y_pred = {y_test_pred[:5]}')

# Predict tomorrow's price for the last point in the test set
last_point_features = X_test.iloc[-1, :]  # Assuming the last row in X_test
last_point_prediction = xgb_reg.predict(last_point_features.values.reshape(1, -1))
st.write(f"Predicted price for tomorrow: {last_point_prediction[0]}")

if cf == 'AAPL':
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
else:
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
pickle_rick = get_model_for_prediction(cf)
prediction = pickle_rick.predict_proba(data)
#st.write(prediction)

st.write("Based on our model, tomorrow's prediction of stock movement should go...")
st.image(image_for_classification(classify_prediction(prediction)))

