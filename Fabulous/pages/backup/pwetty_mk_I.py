import streamlit as st
import datetime

#Greetings
st.title('Hello there!')
st.image('./He-says-the-thing.webp')
st.write('welcome to the META level. Lets get through this together, shall we?')

cf = st.selectbox('Choose your capitalist fighter!!', ["AAPL","MST","AMZN", "WMT","XOM","MCK","ABC","UNH","TSLA","NFLX"])
min_date = datetime.date(2020, 1, 1) # 4years only, this is a dictatorship, with full censorship!
max_date = datetime.date(2024, 1, 14)
start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

if start_date <= end_date:
    pass
else:
    st.error("Error: End date must be after start date.")

#k = st.slider('How far back to the future you wanna go Doc?', min_value=1, max_value=14) # Set the number of days to predict ahead
k = st.selectbox('How far back to the future you wanna go Doc?', [3,5,7,10])


st.color_picker('Pick a color. Don´t look so sus, trust me yalla!') # you rly should not.
st.text("Interessting choices. Shall we boogie?")

st.snow()
#st.camera_input('smile F***Face')


def main_page():
    st.markdown("Hello There")
    st.sidebar.markdown("Hello There")

def Boosting():
    st.markdown("Boost")
    st.sidebar.markdown("Boost")

def page3():
    st.markdown("Fish Memory")
    st.sidebar.markdown("Fish Memory")

page_names_to_funcs = {
    "Hello There": main_page,
    "Boosting": Boosting,
    "Fish Memory": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


