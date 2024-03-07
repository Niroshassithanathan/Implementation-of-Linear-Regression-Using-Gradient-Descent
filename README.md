# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Initialize weights randomly. 

2.Compute predicted values. 

3.Compute gradient of loss function. 

4.Update weights using gradient descent.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: NIROSHA S
RegisterNumber: 212222230097
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term 
  X = np.c_[np.ones(len(X1)), X1]
  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (X).dot(theta).reshape(-1, 1)
    errors = (predictions - y).reshape(-1,1)
    theta -= learning_rate* (1 / len(X1)) * X.T.dot(errors)
  return theta
data = pd.read_csv('50_Startups.csv', header=None)
print(data.head())
# Assuming the last column is your target variable 'y' and the preceding column 
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression(X1_Scaled, Y1_Scaled)
# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/Niroshassithanathan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121418437/5559be51-2493-4206-b343-077e6a648b22)

![image](https://github.com/Niroshassithanathan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121418437/97afb6ef-1bd6-459d-bbcd-016086f868cc)

![image](https://github.com/Niroshassithanathan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121418437/f6dba10f-89c7-49e9-adca-6af1fc7000de)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
