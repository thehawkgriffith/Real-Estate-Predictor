# Real Estate Price Predictor using Linear Regression model v0.2.3
# Created by Shahbaz Khan, August 17th, 2018
import numpy as np
import pandas as pd
import seaborn as sns
import Tkinter as tk
import matplotlib.pyplot as plt

# Importing the CSV file
houses = pd.read_csv('housingdata.csv')

# Importing the method to split testing and training data under Supervised ML
from sklearn.cross_validation import train_test_split

# Importing the Scikit-Learn Linear Regression Model
from sklearn.linear_model import LinearRegression

# Preprocessing data into Feeding and Prediction Data Frames
X = houses[['RM', 'AGE', 'DIS']]
y = houses['MEDV']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 115)

# Training the Model
lm = LinearRegression()
lm.fit(X_train, y_train)
coeff = lm.coef_
coeff_df = pd.DataFrame()

# Gathering Predictions and Residue
predictions = lm.predict(X_test)
residue = y_test - predictions
sns.distplot(residue)
plt.show()


# Application
print('Hello there, to find out the approximate value of the house you are looking for, please answer the following questions to the best of your knowledge.')
print('What are the number of rooms?')
rooms = input()
print('What is the age of the building?')
age = input()
print('What is the distance from city?')
dist = input()

data = np.array([rooms, age, dist])
user_df = pd.DataFrame(data, ['RM', 'AGE', 'DIS']).transpose()

# Prediciting off User given Data
pred = lm.predict(user_df)
res = pred[0] * 10000
print('This Real Estate is going to cost you around $',int(res))
k = input()