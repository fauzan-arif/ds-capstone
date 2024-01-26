from __future__ import division
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Conv1D,Flatten,MaxPooling1D,LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
#from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from sklearn.linear_model import LinearRegression
from helper import save_model, load_model


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

from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load and prepare data
df = load_and_prepare(ticker='AMZN', start_date='2020-01-01', end_date='2024-01-01')

# Select features and target
features = df[['Open', 'High', 'Low', 'Close', 'Volume','alpha', 'momentum_stoch_signal']]
target_class = df['target_class']
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'momentum_stoch_signal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_class, test_size=0.2, random_state=42)
# Define XGBoost model
model = XGBClassifier()
model.feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'momentum_stoch_signal']

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get cross-validation results
cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

# Print average accuracy across folds
print("Average Accuracy: %.2f%%" % (cv_results.mean() * 100))

# Fit the model on the entire training set
model.fit(X_train, y_train)

xgb_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'momentum_stoch_signal']
X_test.columns = xgb_features

# Get the underlying Booster object from the trained model
booster = model.get_booster()

# Set feature names on the Booster
booster.feature_names = xgb_features

# Predictions on the test set
y_pred = model.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))



y_pred_proba = model.predict_proba(X_test)

threshold = 0.50  # Adjust this threshold based on your needs
y_pred_threshold = (y_pred_proba[:, 1] > threshold).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_threshold))

from sklearn.model_selection import GridSearchCV


# Define hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.8, 1.0],
}

# save model
save_model(model, 'model_AMZN.pickle')
# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get best model
best_model = grid_search.best_estimator_

# Predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# save model
save_model(best_model, 'grid_model_AMZN.pickle')
