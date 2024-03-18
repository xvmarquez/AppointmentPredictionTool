import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime, timedelta
import time 

# Load the predictor model from a pickle file
model = pickle.load(open('model.pkl', 'rb'))

# Load the encoder dictionary from a pickle file
with open('encoder.pkl', 'rb') as pkl_file:
    encoder_dict = pickle.load(pkl_file)


def encode_features(df, encoder_dict):
    # For each categorical feature, apply the encoding
    category_col = ['BOOK_DATE', 'ZIPCODE', 'CLINIC', 'APPT_TYPE_STANDARDIZE',
       'ETHNICITY_STANDARDIZE', 'RACE_STANDARDIZE']
    
    # Convert date columns to string before encoding
    if 'BOOK_DATE' in df.columns:
        df['BOOK_DATE'] = df['BOOK_DATE'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else x)


    for col in category_col:
        if col in encoder_dict: 
            le = LabelEncoder()
            le.classes_ = np.array(encoder_dict[col], dtype=object)  # Load the encoder classes for this column

            # Handle unknown categories by using 'transform' method and a lambda function
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            df[col] = le.transform(df[col])

    # Encode the IS_REPEAT column
    df['IS_REPEAT'] = df['IS_REPEAT'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

def main():
    
    st.markdown("""
    <style>
        .centered-text {
            text-align: center;
            font-size: 24px; 
            font-weight: bold; 
        }
    </style>
    <div class="centered-text">CHLA Resource</div>
""", unsafe_allow_html=True)

    
    #st.title("CHLA Resource - Predictor Tool")
    html_temp = """
    <div style="background:#22333b ;padding:10px">
    <h2 style="color:white;text-align:center;">Appointment No Show Prediction App </h2>
    </div>
    """  

    
    st.markdown(html_temp, unsafe_allow_html = True)


    st.info("Please fill in the details of the appointment to predict if the patient will attend")

    # Date input for the appointment
    today = datetime.today()

    # First Date Inputs
    booking_date = st.date_input("Booking Date", min_value= None, value=today)
    if booking_date < datetime.today().date() - timedelta(days=365):
        st.error("Booking Date is too far in the past. Please enter a date within the last year.")
    appointment_date = st.date_input("Appointment Date", min_value=datetime.today())
    
    #appointment_time = st.time_input("Appointment Time")
    # Validate the time input
    #if not (7 <= appointment_time.hour <= 16):
        #st.error("Please select a time between 07:00 and 16:00.")
        #return  # Early return to prevent proceeding with invalid input
    
    LEAD_TIME = (appointment_date - booking_date).days
    if LEAD_TIME < 0:
        st.error("Appointment date cannot be before booking date.")
        return

    def get_week_of_month(year, month, day):
        first_day = datetime.date(year, month, 1)
        first_day_weekday = first_day.weekday()
        if first_day_weekday == 6:
            first_day_weekday = -1
        return ((day + first_day_weekday) // 7) + 1
    
    # Calculating the required features from the appointment date and time
    DAY_OF_WEEK = appointment_date.weekday() + 1  # Monday is 0 and Sunday is 6
    if not (0 <= DAY_OF_WEEK <= 5):
        st.error("Clinics are only open Monday to Saturday, please correct the appointment date.")
        return
    
    #NUM_OF_MONTH = appointment_date.month
    #HOUR_OF_DAY = appointment_time.hour
    WEEK_OF_MONTH = get_week_of_month(appointment_date.year, appointment_date.month, appointment_date.day)
    CLINIC = st.selectbox("Clinic Name", ['ARCADIA CARE CENTER', 'BAKERSFIELD CARE CLINIC', 'ENCINO CARE CENTER', 'SANTA MONICA CLINIC', 'SOUTH BAY CARE CENTER', 'VALENCIA CARE CENTER'])
    ZIPCODE = st.text_input("Zipcode",'00000')
    if len(ZIPCODE) != 5 or not ZIPCODE.isdigit():
        st.error("Please enter a valid 5-digit Zipcode.")
    #APPT_NUM = st.number_input("Number of Previous Appointments",0)
    #TOTAL_NUMBER_OF_CANCELLATIONS = st.number_input("Number of Cancellations",0)
    TOTAL_NUMBER_OF_NOT_CHECKOUT_APPOINTMENT = st.number_input("Number of Not Checked-Out Appointments",0, help="Number of appointments where the patient did not check out correctly.") 
    if TOTAL_NUMBER_OF_NOT_CHECKOUT_APPOINTMENT > 50:  
        st.error("The number of Not Checked-Out Appointments seems too high.")
    TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT = st.number_input("Number of Successful Appointment",0)
    if TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT > 200:
        st.error("The number of Successful Appointments seems too high.")
    LEAD_TIME = LEAD_TIME
    #TOTAL_NUMBER_OF_RESCHEDULED = st.number_input("Number of Rescheduled Appointment",0)
    TOTAL_NUMBER_OF_NOSHOW = st.number_input("Number of No-Shows on record",0)
    if TOTAL_NUMBER_OF_NOSHOW > 100:
        st.error("The number of No-Shows seems too high.")
    #AGE = st.number_input("Age of Patient",0)
    #if AGE < 0 or AGE > 120:
    #   st.error("Please enter a valid age.") 
    IS_REPEAT = st.checkbox("Repeat Patient") 
    ETHNICITY_STANDARDIZE = st.selectbox("Ethnicity",['Hispanic', 'Non-Hispanic', 'Others'])
    APPT_TYPE_STANDARDIZE = st.selectbox("Appointment Type",['Follow-up', 'New', 'Others'])
    RACE_STANDARDIZE =  st.selectbox("Race",['African', 'Asian','European','Middle Eastern', 'North American', 'Other','South American'])

    

    if st.button("Predict"):

        data = { 
        'BOOK_DATE': booking_date,
        'ZIPCODE': ZIPCODE,
        'CLINIC': CLINIC, 
        'DAY_OF_WEEK': DAY_OF_WEEK,
        'WEEK_OF_MONTH': WEEK_OF_MONTH,
        'LEAD_TIME': LEAD_TIME, 
        'IS_REPEAT':IS_REPEAT, 
        'APPT_TYPE_STANDARDIZE':APPT_TYPE_STANDARDIZE,
       'TOTAL_NUMBER_OF_NOT_CHECKOUT_APPOINTMENT':int(TOTAL_NUMBER_OF_NOT_CHECKOUT_APPOINTMENT),
       'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT':int(TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT), 
        'TOTAL_NUMBER_OF_NOSHOW':int(TOTAL_NUMBER_OF_NOSHOW), 
        'ETHNICITY_STANDARDIZE':ETHNICITY_STANDARDIZE, 
        'RACE_STANDARDIZE':RACE_STANDARDIZE
       }
        
        # print(data)
        # Convert the data into a DataFrame for easier manipulation
        df = pd.DataFrame([data])
        
        # Encode the categorical columns
        df = encode_features(df, encoder_dict)

        with st.spinner('Processing... Please wait'):
            time.sleep(2)
            # Convert the DataFrame into a list of values and make a prediction
            features_list = df.values
            prediction = model.predict(features_list)
            

        output = int(prediction[0])
        
        if output == 1:
            message = 'The patient will NOT attend.'
            st.error(message)  # Using st.error to highlight a potential issue (patient not attending)
        else:
            message = 'The patient will attend.'
            st.success(message)

if __name__=='__main__':
    main()

