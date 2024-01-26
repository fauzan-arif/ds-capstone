import streamlit as st
import datetime
from xgboost_page import xgboost_predict  # Assume you have a function in xgboost_page.py
from lstm_page import lstm_predict  # Assume you have a function in lstm_page.py

# Greetings
st.title('Hello there!')
st.image('./He-says-the-thing.webp')
st.write('Welcome to the META level. Let us embark on this journey together!')

# Sidebar enchantment
with st.sidebar:
    st.title('Configuration')

    cf = st.selectbox('Choose your capitalist fighter!!', ["AAPL", "MST", "AMZN", "WMT", "XOM", "MCK", "ABC", "UNH", "TSLA", "NFLX"])
    
    min_date = datetime.date(2020, 1, 1)  # 4 years only, this is a dictatorship, with full censorship!
    max_date = datetime.date(2024, 1, 14)
    
    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

    if start_date <= end_date:
        pass
    else:
        st.error("Error: End date must be after start date.")

    k = st.selectbox('How far back to the future you wanna go Doc?', [3, 5, 7, 10])
    slected_model = st.selectbox("Chooose your weaponery", ["XGBoost","LSTM"])
    color = st.color_picker('Pick a color. Don’t look so sus, trust me yalla!')  # you really should not.

# Continue the adventure
st.text("Interesting choices. Shall we boogie?")
st.snow()
#st.camera_input('smile F***Face')

# The magical button to unleash the predictions!
if st.button('Predict the Future!'):
    # Invoke the chosen prediction model based on the sidebar selections
    if slected_model == "XGBoost":
        xgboost_predict(cf, start_date, end_date, k, color)
    elif selected_model == "LSTM":
        lstm_predict(cf, start_date, end_date, k, color)
    else:
        st.error("Unknown model selected! A wizard never chooses the wrong path.")