# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
loaded_model = pickle.load(open('E:/Diabetes_prediction_webapp/trained_model.sav','rb'))

input_data=(3,82,70,0,0,21.1,0.389,25)

input_numpy=np.asarray(input_data)

input_data_reshape=input_numpy.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshape)

if(prediction[0]==0):
    print("The person is not diabetic !")
else:
    print("The person is diabetic !")