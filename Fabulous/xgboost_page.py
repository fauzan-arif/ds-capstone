import streamlit as st
import numpy as np
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

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

import yfinance as yf


#def run_xgboost_analysis(tickers, start_date, end_date, k, color):
 #   for ticker in tickers:
  #      xgboost_predict(ticker, start_date, end_date, k, color)

def xgboost_predict(cf, start_date, end_date, k, color):
    #import streamlit as st
    #import requestreamlits
    import streamlit as st
    import numpy as np
    import xgboost as xgb
    import matplotlib.pyplot as plt
    from xgboost import plot_importance, plot_tree
    from sklearn.metrics import (
        mean_squared_error,
        accuracy_score,
        classification_report
        )
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

    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=DeprecationWarning)

    import yfinance as yf

    #Buckle up Buckaroo, time for some magic  
    ticker_symbol = cf
    amazon_stock = yf.download(ticker_symbol, start=start_date, end=end_date)
    # Display the data
    import streamlit as st
    #st.dataframe(amazon_stock)#.head()

    #--

    import pandas as pd
    amazon_stock.index = pd.to_datetime(amazon_stock.index)
    # Reset the index to make "Date" a regular column
    amazon_stock.reset_index(drop=False, inplace=True)

    #--
    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Ohlc(x=amazon_stock.Date,
                          open=amazon_stock.Open,
                          high=amazon_stock.High,
                          low=amazon_stock.Low,
                          close=amazon_stock.Close,
                          name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock.Volume, name='Volume'), row=2, col=1)

    fig.update(layout_xaxis_rangeslider_visible=False)
#    fig.show()
    #fig.write_image(file='fig', format='png')
    #st.image("fig.png")

    #st.pyplot(fig)


    #--

    # Creating technical indicators - exponential moving averages and short moving averages
    amazon_stock['EMA_9'] = ta.ema(amazon_stock['Close'], length=9, fillna=True).shift()
    amazon_stock['SMA_5'] = ta.sma(amazon_stock['Close'], length=5, fillna=True).shift()
    amazon_stock['SMA_10'] = ta.sma(amazon_stock['Close'], length=10, fillna=True).shift()
    amazon_stock['SMA_15'] = ta.sma(amazon_stock['Close'], length=15, fillna=True).shift()
    amazon_stock['SMA_30'] = ta.sma(amazon_stock['Close'], length=30, fillna=True).shift()
    #--

    

    # Creating relative strength index
    amazon_stock['RSI'] = ta.rsi(amazon_stock['Close'], length=14, fillna=True)

    # Plot RSI
    fig = go.Figure(go.Scatter(x=amazon_stock['Date'], y=amazon_stock['RSI'], name='RSI'))
#    fig.show()
    #--
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

    # Removing unnecessary features from train and test df
    train_df = train_df.drop(columns=['Open', 'High', 'Low','Date', 'Adj Close', 'Volume', 'Close', 'MACD_diff', 'ROC', 'OBV'])  # Date,  Close, Volume, OpenInt)
    test_df = test_df.drop(columns=['Open', 'High', 'Low','Date', 'Adj Close', 'Volume', 'Close', 'MACD_diff', 'ROC', 'OBV'])  # Date,  Close, Volume, OpenInt)

    # Making copies for scaling
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    # Extracting target and features for training
    y_train = train_df['close_tmr'].copy()
    X_train = train_df.drop(['close_tmr'], axis=1)

    # Extracting target and features for testing
    y_test = test_df['close_tmr'].copy()
    X_test = test_df.drop(['close_tmr'], axis=1)


    from xgboost import XGBRegressor
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
    print(f'Test RMSE: {rmse_test}')
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # Calculate MAPE on the test set
    mape_test = calculate_mape(y_test, y_test_pred)
    print(f'Test MAPE: {mape_test:.2f}%')
    

    import numpy as np
    print(f'y_true = {np.array(y_test)[:5]}')
    print(f'y_pred = {y_test_pred[:5]}')

    from sklearn.metrics import accuracy_score, classification_report

    k = 5 # Set the number of days to predict ahead

    # Convert regression predictions to binary labels (1 for price up, 0 for price down)
    predicted_direction = np.where(y_test_pred[k:] > y_test_pred[:-k], 1, 0)

    # Create true binary labels based on the actual price movements
    true_direction = np.where(y_test.values[k:] > y_test.values[:-k], 1, 0)


    # Calculate accuracy and other classification metrics
    accuracy = accuracy_score(true_direction, predicted_direction)
    classification_report_result = classification_report(true_direction, predicted_direction)


    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report_result)
    plot_importance(xgb_reg)
    

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Plot actual closing prices
    plt.plot(y_test.index, y_test.values, label='Actual Closing Price')

    # Plot predicted closing prices
    plt.plot(y_test.index, y_test_pred, label='Predicted Closing Price')

    plt.xlabel('Days')
    plt.ylabel('Closing Price')
    plt.title('Actual vs Predicted Closing Prices')
    plt.legend()
    plt.show()

    st.write(f"XGBoost predictions for {cf} from {start_date} to {end_date} with {k} days into the future.")
    st.button
    st.balloons
    #xgb_reg = None  # Initialize with appropriate XGBoost model
    #y_test_pred = None  # Initialize with appropriate predictions
    #xgboost_predict(cf, start_date, end_date, k, color)
#st.stop
    