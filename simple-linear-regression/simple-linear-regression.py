# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# splitting the data into train and test set
x_train, x_test, y_train, y_test = train_test_split(x,  y, test_size=0.2, random_state=1)

# training the simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test set results
y_pred = regressor.predict(x_test)

# visualising the training set results
plt.scatter(x_train, y_train, color='red') # scatter plot
plt.plot(x_train, regressor.predict(x_train), color='blue') # line plot
plt.title('Salary vs Experience (Train Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualising the test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, regressor.predict(x_test), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()