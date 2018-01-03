import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt


#Import dataset
dataset = pd.read_csv('titanic.csv')


#Drop the unnecessary columnns by their column indices
dataset.drop(dataset.columns[[3,8,10]],axis = 1,inplace = True)


#Calculate the Nan values  in dataset. This will show you total Nan values of each category if you're using Spyder IDE
print (dataset.isnull().sum())


#If there are any then fill them with median or any other strategy
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())


#Confirm again for Nan
print (dataset['Age'].isnull().sum())


#Segregate into dependent(Target) and independent(Other than target)
y = dataset['Survived'].values
x = dataset.drop(dataset.columns[1] , axis = 1)
x = x.drop(x.columns[7] , axis = 1)


#Feature Scaling
from sklearn.preprocessing import LabelEncoder
sc = LabelEncoder()
x['Sex'] = sc.fit_transform(x['Sex'])


#Split into training and test data
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test =  train_test_split(x , y , test_size = 0.2, random_state = 42)

#Use the required model. I have used Decision Tree regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 42)

#Fit the regressor object
regressor.fit(x_train ,  y_train)

#Predict the values for the test set
y_pred = regressor.predict(x_test)

#Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test , y_pred)
