from __future__ import division
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import streamlit as st 
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Conv1D,Flatten,MaxPooling1D,LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
#from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Identify stock data to grab by ticker
ticker = 'AAPL'

start_date=datetime.datetime(2016,1,6)
end_date=datetime.datetime(2021,1,5)

import pandas as pd 
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from sklearn.linear_model import LinearRegression

def load_and_prepare(ticker, start_date, end_date):
    # Fetch historical stock data
    df = yf.download(ticker, start=start_date , end=end_date)

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
    merged_data = add_all_ta_features(merged_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    # Calculate target and target class
    merged_data['target'] = ((merged_data['Close'] - merged_data['Open']) / merged_data['Open']) * 100
    merged_data['target'] = merged_data['target'].shift(-1)
 
    merged_data['target_class'] = np.where(merged_data['target'] < 0, 0, 1)
    
    merged_data['target_next_close'] = merged_data['Close'].shift(-1)

    # Drop rows with missing values
    merged_data.dropna(inplace=True)

    return merged_data

df = load_and_prepare('AAPL', "2010-01-01" , "2024-01-01")


df

df["rapp"]=df["Close"].divide(df['Close'].shift(1)) # Should be the close of the previous close


df["mv_avg_short"]= df["Close"].rolling(window=5).mean()
df["mv_avg_long"]= df["Close"].rolling(window=50).mean()



df=df.iloc[50:,:] # WARNING: DO IT JUST ONE TIME!
st.write(df.index)


len(df)



df = df.reset_index()



from sklearn.model_selection import train_test_split

# Assuming 'df' is your DataFrame
train, test = train_test_split(df, test_size=600, shuffle=False)

# Display the shapes of the resulting DataFrames
st.write("Train shape:", train.shape)
st.write("Test shape:", test.shape)



# This function returns the total percentage gross yield and the annual percentage gross yield

def yield_gross(df,v):
    prod=(v*df["rapp"]+1-v).prod()
    n_years=len(v)/252
    return (prod-1)*100,((prod**(1/n_years))-1)*100



def create_window(data, window_size = 1):    
    data_s = data.copy()
    for i in range(window_size):
        data = pd.concat([data, data_s.shift(-(i + 1))], axis = 1)
        
    data.dropna(axis=0, inplace=True)
    return(data)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
scaler = MinMaxScaler(feature_range=(0, 1))
dg = pd.DataFrame(scaler.fit_transform(df[["High", "Low", "Open", "Close", "Volume", "mv_avg_short", "mv_avg_long"]].values))
dg0 = dg[[0, 1, 2, 3, 4, 5]]

window = 4
dfw = create_window(dg0, window)

X_dfw = np.reshape(dfw.values, (dfw.shape[0], window + 1, 6))

y_dfw = np.array(dg[6][window:])  # The Fix

# Adjust mtest based on the new shape of your data
mtest = 600

X_trainw, X_testw, y_trainw, y_testw = train_test_split(X_dfw, y_dfw, test_size=mtest, shuffle=False)

def model_lstm(window, features):
    model = Sequential()
    model.add(LSTM(300, input_shape=(window, features), return_sequences=True))
    model.add(Dropout(0.2))  # Add dropout layer with a dropout rate (fraction of input units to drop)
    model.add(LSTM(200, input_shape=(window, features), return_sequences=False))
    model.add(Dropout(0.2))  # Add dropout layer with a dropout rate
    model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))  # Add dropout layer with a dropout rate
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
    model.compile(loss='mse', optimizer='adam')

    return model

model = model_lstm(window + 1, 6)
history = model.fit(X_trainw, y_trainw, epochs=100, batch_size=200, validation_data=(X_testw, y_testw), \
                    verbose=1, callbacks=[], shuffle=False)  # Batch size should be no more than the square root of the # of training rows

# Plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig("plot_I.png")
st.image("plot_I.png")


y_pr=model.predict(X_trainw)

import matplotlib.pyplot as plt
import matplotlib.style as style

# Set the style to a dark theme
style.use('dark_background')

# Assuming y_trainw and y_pr are your actual and predicted values
plt.figure(figsize=(30, 10))
plt.plot(y_trainw, label='Actual', color='limegreen', linewidth=2)  # Customize the color and linewidth as needed
plt.plot(y_pr, label='Prediction', color='gold', linewidth=2)  # Customize the color and linewidth as needed

# Set title and labels with light text for better visibility
plt.title('Actual Moving Average vs Predicted', fontsize=25, color='white')
plt.xlabel('Data Points', fontsize=20, color='white')
plt.ylabel('Value', fontsize=20, color='white')

# Add legend with light text
plt.legend(fontsize=20, loc='upper right', frameon=False, facecolor='none', edgecolor='none', labelcolor='white')

# Add grid lines
plt.grid(axis='both', color='gray', linestyle='--', linewidth=0.5)

# Show the plot
plt.savefig("plot_II.png")
st.image("plot_II.png")

# Check data shapes
st.write("y_testw shape:", y_testw.shape)
st.write("y_pr shape:", y_pr.shape)

# Reshape y_pr to 1D array
y_pr_1d = y_pr.ravel()

# Convert to pandas DataFrame if not already
y_testw_df = pd.DataFrame({'true_labels': y_testw})
y_pr_df = pd.DataFrame({'predicted_labels': y_pr_1d})

# Concatenate data along columns for easier comparison
combined_df = pd.concat([y_testw_df, y_pr_df], axis=1)

# Check for missing values
st.write("Missing values in combined DataFrame:")
st.write(combined_df.isnull().sum())

# Check if indices are aligned
st.write("Index alignment check:")
st.write(combined_df.index.equals(y_testw_df.index) and combined_df.index.equals(y_pr_df.index))

import pandas as pd

# Check data shapes
st.write("y_testw shape:", y_testw.shape)
st.write("y_pr shape:", y_pr.shape)

# Reshape y_pr to 1D array
y_pr_1d = y_pr.ravel()

# Convert to pandas DataFrame if not already
y_testw_df = pd.DataFrame({'true_labels': y_testw})
y_pr_df = pd.DataFrame({'predicted_labels': y_pr_1d})

# Concatenate data along columns for easier comparison
combined_df = pd.concat([y_testw_df, y_pr_df], axis=1)

# Check for missing values
st.write("Missing values in combined DataFrame:")
st.write(combined_df.isnull().sum())

# Check if indices are aligned
st.write("Index alignment check:")
st.write(combined_df.index.equals(y_testw_df.index) and combined_df.index.equals(y_pr_df.index))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Assuming 'df' is your DataFrame
scaler = MinMaxScaler(feature_range=(0, 1))
dg = pd.DataFrame(scaler.fit_transform(df[["High", "Low", "Open", "Close", "Volume", "mv_avg_short", "mv_avg_long"]].values))
dg0 = dg[[0, 1, 2, 3, 4, 5]]

window = 4
dfw = create_window(dg0, window)

X_dfw = np.reshape(dfw.values, (dfw.shape[0], window + 1, 6))

y_dfw = np.array(dg[6][window:])  # The Fix

# Create TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)  # You can adjust the number of splits as needed

mse_scores = []  # to store MSE for each fold
r2_scores = []   # to store R2 scores for each fold
mae_scores = []  # to store MAE for each fold

# Iterate over train-test splits
for train_index, test_index in tscv.split(X_dfw):
    X_trainw, X_testw = X_dfw[train_index], X_dfw[test_index]
    y_trainw, y_testw = y_dfw[train_index], y_dfw[test_index]

    model = model_lstm(window + 1, 6)
    history = model.fit(X_trainw, y_trainw, epochs=50, batch_size=200, validation_data=(X_testw, y_testw), \
                        verbose=1, callbacks=[], shuffle=False)

    # Make predictions on the test set
    y_pred = model.predict(X_testw)

    # Evaluate the model
    mse = mean_squared_error(y_testw, y_pred)
    r2 = r2_score(y_testw, y_pred)
    mae = mean_absolute_error(y_testw, y_pred)
    mse_scores.append(mse)
    r2_scores.append(r2)
    mae_scores.append(mae)

# Print average scores across folds
st.write(f'Average Mean Squared Error (MSE): {np.mean(mse_scores)}')
st.write(f'Average R-squared (R2) Score: {np.mean(r2_scores)}')
st.write(f'Average Mean Absolute Error (MAE): {np.mean(mae_scores)}')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("plot_III.png")
st.image("plot_III.png")


#y_predicted = model.predict(X_testw)
style.use('dark_background')
plt.figure(figsize=(15, 8))
plt.plot(y_testw, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted Moving Averages(Test)')
plt.xlabel('Data Points')
plt.ylabel('Value')
plt.legend()
plt.savefig("plot_IV.png")
st.image("plot_IV.png")




from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_testw, y_pred)
r2 = r2_score(y_testw, y_pred)

st.write(f'Mean Squared Error (MSE): {mse}')
st.write(f'R-squared (R2) Score: {r2}')



# Evaluate the model on the test set
loss = model.evaluate(X_testw, y_testw)
st.write(f'Test Loss (MSE): {loss}')

from sklearn.metrics import mean_squared_error

# Assuming 'model' is your trained model
y_pred = model.predict(X_testw)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_testw, y_pred)
st.write(f'Mean Squared Error (MSE): {mse}')



# Use the last window days of data to predict the next day's price
last_window_data = X_dfw[-1:]
predicted_price = model.predict(last_window_data)


predicted_price



y_pred = model.predict(X_testw)

import matplotlib.pyplot as plt
import matplotlib.style as style

# Set the style to a dark theme
style.use('dark_background')

# Assuming y_testw and y_pred are your actual and predicted values
plt.figure(figsize=(15, 8))
plt.plot(y_testw, label='Actual', color='yellow')  # Customize the color as needed
plt.plot(y_pred, label='Predicted', color='cyan')  # Customize the color as needed

# Set title and labels with light text for better visibility
plt.title('Actual vs Predicted Values', color='white')
plt.xlabel('Data Points', color='white')
plt.ylabel('Value', color='white')

# Add legend with light text
plt.legend(fontsize='small', loc='upper right', frameon=False, facecolor='none', edgecolor='none', labelcolor='white')

# Show the plot
plt.savefig("plot_V.png")
st.image("plot_V.png")



X_testw


