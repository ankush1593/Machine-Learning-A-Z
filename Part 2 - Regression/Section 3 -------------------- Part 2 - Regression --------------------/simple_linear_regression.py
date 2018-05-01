#Importing basic libraries
import pandas as pd
import matplotlib.pyplot as plt

#Getting dataset and extracting X and Y
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitiing training and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#Linear Regression model and fitting X_train in it
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predictions of test data
y_pred = regressor.predict(X_test)

#Visualization of Train case
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualization of Train case
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()