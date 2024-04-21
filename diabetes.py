
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_set=pd.read_csv("DataSets\\diabetes.csv")

diabetes_set.head()
diabetes_set.shape
diabetes_set.describe()
diabetes_set['Outcome'].value_counts()
diabetes_set.groupby('Outcome').mean()
X=diabetes_set.drop(columns= 'Outcome',axis=1)
Y=diabetes_set['Outcome']
print(X)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
#test_size means the percentage of data to be trained i.e. 0.2 means 20%
print(X.shape,X_train.shape,X_test.shape)
# TRAINING THE MODEL ----> SVM(Support Vector Machine)
model = svm.SVC(kernel='linear')
model.fit(X_train,Y_train)
# MODEL EVALUATION
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data is:',training_data_accuracy)
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy on training data is:',test_data_accuracy)

# MAKING A PREDICTIVE SYSTEM 
input_data=(3,82,70,0,0,21.1,0.389,25)
input_numpy=np.asarray(input_data)
input_data_reshape=input_numpy.reshape(1,-1)
prediction = model.predict(input_data_reshape)
if(prediction[0]==0):
    print("The person is not diabetic !")
else:
    print("The person is diabetic !")



