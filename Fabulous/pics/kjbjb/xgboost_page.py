import streamlit as st
import pandas as pd

def xgboost_predict(cf, start_date, end_date, k, color):
    #import streamlit as st
    
    #import requestreamlits
    import plotly as py
    import plotly.graph_objects as go
    import yfinance as yf
    #import sqlalchemy 
    import datetime
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from xgboost import XGBRegressor
    # Chart drawing
    import plotly as py
    import plotly.io as pio
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    # Mute sklearn warnings
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=DeprecationWarning)
    import yfinance as yf 
    from sklearn.preprocessing import MinMaxScaler  
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import accuracy_score, classification_report
    from xgboost import plot_importance, plot_tree
    #Buckle up Buckaroo, time for some magic
    # Define the ticker symbol for S&P 500 (SPY is an ETF that tracks the S&P 500)
    # min and max today and today -4year    
    ticker_symbol = cf


    # Use yfinance to download the data
    amazon_stock = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Display the data
    st.dataframe(amazon_stock)#.head()

    import pandas as pd

    amazon_stock.index = pd.to_datetime(amazon_stock.index)

    # Reset the index to make "Date" a regular column
    amazon_stock.reset_index(drop=False, inplace=True)


    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Ohlc(x=amazon_stock.Date,
                        open=amazon_stock.Open,
                        high=amazon_stock.High,
                        low=amazon_stock.Low,
                        close=amazon_stock.Close,
                        name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock.Volume, name='Volume'), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)
    #fig.update(layout_xaxis_rangeslider_visible=False)
    #fig.show()
    #fig.write_image('fig_u_I.png')

    #st.image("fig_u_I.png")
    #st.warning('WTF?!')
    #st.error('u fcked up.')
    st.toast('Wallah Magie!!')
    df_close = amazon_stock[['Date', 'Close']].copy()
    df_close = df_close.set_index('Date')
    st.text("Behold, df_close:")
    df_close.head()

    if len(df_close) >= 730:
        decomp = seasonal_decompose(df_close, period=365)
    
    
        fig = decomp.plot()
        fig.set_size_inches(20, 8)

        amazon_stock['EMA_9'] = amazon_stock['Close'].ewm(9).mean().shift()
        amazon_stock['SMA_5'] = amazon_stock['Close'].rolling(5).mean().shift()
        amazon_stock['SMA_10'] = amazon_stock['Close'].rolling(10).mean().shift()
        amazon_stock['SMA_15'] = amazon_stock['Close'].rolling(15).mean().shift()
        amazon_stock['SMA_30'] = amazon_stock['Close'].rolling(30).mean().shift()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock.EMA_9, name='EMA 9'))
        fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock.SMA_5, name='SMA 5'))
        fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock.SMA_10, name='SMA 10'))
        fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock.SMA_15, name='SMA 15'))
        fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock.SMA_30, name='SMA 30'))
        fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock.Close, name='Close', opacity=0.2))
        #fig.show()
    else:
        st.error("Not enough observations for seasonal decomposition. Please ensure at least 730 observations.")


    def relative_strength_idx(df, n=14):
        close = amazon_stock['Close']
        delta = close.diff()
        delta = delta[1:]
        pricesUp = delta.copy()
        pricesDown = delta.copy()
        pricesUp[pricesUp < 0] = 0
        pricesDown[pricesDown > 0] = 0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp / rollDown
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    amazon_stock['RSI'] = relative_strength_idx(amazon_stock).fillna(0)

    fig = go.Figure(go.Scatter(x=amazon_stock.Date, y=amazon_stock.RSI, name='RSI'))
    #fig.show()


    EMA_12 = pd.Series(amazon_stock['Close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(amazon_stock['Close'].ewm(span=26, min_periods=26).mean())
    amazon_stock['MACD'] = pd.Series(EMA_12 - EMA_26)
    amazon_stock['MACD_signal'] = pd.Series(amazon_stock.MACD.ewm(span=9, min_periods=9).mean())

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock.Close, name='Close'), row=1, col=1)
    fig.add_trace(go.Scatter(x=amazon_stock.Date, y=EMA_12, name='EMA 12'), row=1, col=1)
    fig.add_trace(go.Scatter(x=amazon_stock.Date, y=EMA_26, name='EMA 26'), row=1, col=1)
    fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock['MACD'], name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=amazon_stock.Date, y=amazon_stock['MACD_signal'], name='Signal line'), row=2, col=1)
    #fig.show()

    amazon_stock['close_tmr'] = amazon_stock['Close'].shift(-1)
    amazon_stock['close_today'] = amazon_stock['Close']


    amazon_stock['OBV'] = (amazon_stock['Volume'] * ((amazon_stock['close_today'] - amazon_stock['close_today'].shift(1)) > 0)).cumsum()
    amazon_stock['ROC'] = amazon_stock['Close'].pct_change() * 100  # Calculate percentage change

    amazon_stock



    amazon_stock = amazon_stock.iloc[33:] # Because of moving averages and MACD line
    amazon_stock = amazon_stock[:-1]      # Because of shifting close price

    amazon_stock.index = range(len(amazon_stock))
    #st.text("Take a look at this tail!")
    amazon_stock.tail()



    amazon_stock['Year'] = amazon_stock['Date'].dt.year
    amazon_stock['Month'] = amazon_stock['Date'].dt.month
    amazon_stock['Day'] = amazon_stock['Date'].dt.day

    # # Load in sentiment

    #####

    test_size  = 0.15
    valid_size = 0.15

    test_split_idx  = int(amazon_stock.shape[0] * (1-test_size))
    valid_split_idx = int(amazon_stock.shape[0] * (1-(valid_size+test_size)))

    train_df = amazon_stock.loc[:valid_split_idx].copy()
    valid_df = amazon_stock.loc[valid_split_idx+1:test_split_idx].copy()
    test_df = amazon_stock.loc[test_split_idx+1:].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.Date, y=train_df.close_tmr, name='Training'))
    fig.add_trace(go.Scatter(x=valid_df.Date, y=valid_df.close_tmr, name='Validation'))
    fig.add_trace(go.Scatter(x=test_df.Date,  y=test_df.close_tmr,  name='Test'))
    #fig.show()


    drop_cols = ['Date', 'Open', 'Low', 'High', 'Adj Close', 'Volume', 'OBV', 'ROC','Close']  # Date, Open, High, Low, Close, Volume, OpenInt

    # Assuming train_df, valid_df, and test_df are your DataFrames
    train_df = train_df.drop(columns=drop_cols)
    valid_df = valid_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    # Making copies
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()





    # Assuming df is your DataFrame with the features you want to scale
    columns_to_scale = ['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal', 'close_tmr', 'close_today']

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on the selected columns and transform the data
    train_df_scaled[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    test_df_scaled[columns_to_scale] = scaler.transform(test_df[columns_to_scale])

    #Labels and features

    y_train = train_df['close_tmr'].copy()
    X_train = train_df.drop(['close_tmr'], axis=1)  # Drop both 'Close' and 'Adj Close'

    y_train_scaled = train_df_scaled['close_tmr'].copy()
    X_train_scaled = train_df_scaled.drop(['close_tmr'], axis=1)  # Drop both 'Close' and 'Adj Close'

    y_valid = valid_df['close_tmr'].copy()
    X_valid = valid_df.drop(['close_tmr'], axis=1)  # Drop both 'Close' and 'Adj Close'

    y_test  = test_df['close_tmr'].copy()
    X_test  = test_df.drop(['close_tmr'], axis=1)  # Drop both 'Close' and 'Adj Close'

    y_test_scaled  = test_df_scaled['close_tmr'].copy()
    X_test_scaled  = test_df_scaled.drop(['close_tmr'], axis=1)  # Drop both 'Close' and 'Adj Close'

    X_train.isna().sum()
    #y_test


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
    st.text(f'Test RMSE: {rmse_test}')

    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # Calculate MAPE on the test set
    mape_test = calculate_mape(y_test, y_test_pred)
    st.text(f'Test MAPE: {mape_test:.2f}%')

    import numpy as np
    st.text(f'y_true = {np.array(y_test)[:5]}')
    st.text(f'y_pred = {y_test_pred[:5]}')





    # Convert regression predictions to binary labels (1 for price up, 0 for price down)
    predicted_direction = np.where(y_test_pred[k:] > y_test_pred[:-k], 1, 0)

    # Create true binary labels based on the actual price movements
    true_direction = np.where(y_test.values[k:] > y_test.values[:-k], 1, 0)


    # Calculate accuracy and other classification metrics
    accuracy = accuracy_score(true_direction, predicted_direction)
    classification_report_result = classification_report(true_direction, predicted_direction)


    st.text(f"Accuracy: {accuracy}")
    st.text("Classification Report:")
    st.text(classification_report_result)



    plot_importance(xgb_reg)


    st.write(f"XGBoost predictions for {cf} from {start_date} to {end_date} with {k} days into the future.")
    
    st.balloons
