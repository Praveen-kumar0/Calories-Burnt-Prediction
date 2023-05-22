# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:05:57 2023

@author: aa
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# creating a function for prediction
def Calories_Burnt_Prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction


def main():
    
    # giving a title
    st.title("Calories Burnt Prediction Web App")
    
    # getting the input data from the user
    
    Gender = st.number_input('Gender(0 for male, 1 for female) ')
    Age = st.number_input('Age ')
    Height = st.number_input('Height(in cm) ')
    Weight = st.number_input('Weight(in kgs) ')
    Duration = st.number_input('Duration(in min) ')
    Heart_Rate = st.number_input('Heart_Rate(in bpm) ')
    Body_Temp = st.number_input('Body_Temp(in celsius) ')
    
    
    # code for prediction
    result = ''
    
    # creating a button for prediction
    
    if st.button('Calories Burnt Result'):
        result = Calories_Burnt_Prediction([Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp])
        
    st.success(result)
    
    
    
if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
