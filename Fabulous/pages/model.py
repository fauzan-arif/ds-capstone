import streamlit as st
import datetime
import os
import yfinance as yf
#from xgboost_page import xgboost_predict  
#from lstm_page import lstm_predict  
#from XGBClassifier_AAPL import xgboost_AAPL_predict
import pickle
from helper import save_model, load_model
# Greetings
st.title('Wanna see a cool trick?')
#st.image('./pics/grievous.jpeg')
st.write('Configurations to be played with in sidebar! ')
# Sidebar enchantment
''' Make Bar go sideways, multi headers, add price and movement with 1 becoming arrow, snow out, bar/chart, reverse in regressor 
that new data comes first,'''
#st.snow()

with st.sidebar:
    st.title('Configuration')
    cf = st.selectbox('Choose your capitalist fighter!!', ["AAPL", "MSFT", "AMZN", "META", "GOOG"]) 
    period = st.selectbox("How far you wanna go back?", ['1d', '5d', '1mo', 'max'])
    #k = st.selectbox('How far back to the future you wanna go Doc?', [3, 5, 7, 10])
    color = st.color_picker('Pick a color. Don’t look so sus, trust me yalla!')  # you really should not.

# Continue the adventure
#st.text("Interesting choices. Shall we?")
#pickle_rick = load_model('../model_AAPL.pickle')
#prediction = pickle_rick.predict(input)
#st.button('Let's go back to the future Doc!')

from ta import add_all_ta_features
def load_data_from_yfinance(cf, period):
    data = yf.Ticker(cf).history(period=period)
    return data

def classify_prediction(prediction, threshold=0.52):
  return (prediction[:, 1] > threshold).astype(int)[0]
    

def get_model_for_prediction(cf):
    file = os.path.join("pages", "models", f"grid_model_{cf}.pickle")
    return load_model(file)


data = load_data_from_yfinance(cf, period)

if cf == 'AAPL':
    data = data.drop(columns=['Dividends','Stock Splits'])
    pickle_rick = get_model_for_prediction(cf)
    prediction = pickle_rick.predict_proba(data)
    st.write(prediction)
    st.write(classify_prediction(prediction))
elif cf == 'AMZN':
    data = data.drop(columns=['Dividends','Stock Splits'])
    pickle_rick = get_model_for_prediction(cf)
    prediction = pickle_rick.predict_proba(data)
    st.write(prediction)
    st.write(classify_prediction(prediction))

elif cf == 'META':
    data = data.drop(columns=['Dividends','Stock Splits'])
    pickle_rick = get_model_for_prediction(cf)
    prediction = pickle_rick.predict_proba(data)
    st.write(prediction)
    st.write(classify_prediction(prediction))

if cf == 'MSFT':
    data = data.drop(columns=['Dividends','Stock Splits'])
    pickle_rick = get_model_for_prediction(cf)
    prediction = pickle_rick.predict_proba(data)
    st.write(prediction)
    st.write(classify_prediction(prediction))

elif cf == 'GOOG':
    data = data.drop(columns=['Dividends','Stock Splits'])
    pickle_rick = get_model_for_prediction(cf)
    prediction = pickle_rick.predict_proba(data)
    st.write(prediction)
    st.write(classify_prediction(prediction))
    
else:
    st.text(" ")


#st.button('Let's go back to the future Doc!')
    

#st.stop