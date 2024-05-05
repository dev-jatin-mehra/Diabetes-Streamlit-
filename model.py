# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:44:08 2024

@author: jatin
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('./diabetes_trained_model.sav','rb'))

#creating a function for prediction

def diabetes_prediction(input_data):

    # input_numpy=np.asarray(input_data,dtype=float)
    
    # #predicting for one instance
    # input_data_reshape=input_numpy.reshape(1,-1)

    # prediction = loaded_model.predict(input_data_reshape)

    # if(prediction[0]==0):
    #     return 'The person is not diabetic !'
    # else:
    #     return 'The person is diabetic !'

    if not input_data or '' in input_data:
        return "Please provide valid input for all fields."
    
    try:
        input_numpy = np.asarray(input_data, dtype=float)  # Convert input data to floats
        input_data_reshape = input_numpy.reshape(1, -1)
        prediction = loaded_model.predict(input_data_reshape)
        if prediction[0] == 0:
            return 'The person is not diabetic!'
        else:
            return 'The person is diabetic!'
    except ValueError:
        return "Invalid input. Please provide numerical values for all fields."
    
def main():
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getiing input data from user 
    
    Pregnancies = st.text_input('Number of pregnencies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood pressure Value')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('Body-Mass Index Value')
    DiabetesPedigreeFunction = st.text_input('DPF')
    Age = st.text_input('Age Of the person')
    
    #code for prediction 
    diagnosis = ""
    
    #creating a button
    
    if st.button('GENERATE RESULT'):
       diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]) 
       
    st.success(diagnosis)
    
    
    
if __name__=='__main__':
    main()
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       