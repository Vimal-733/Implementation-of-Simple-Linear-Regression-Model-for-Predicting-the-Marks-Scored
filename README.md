# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Take study hours (X) and corresponding marks (Y).
2. Create a linear regression model and fit it using X and Y.
3. Accept new input (hours studied) and predict the marks.
4. Print slope and intercept, and plot the data with the regression line.

## Program:
```
*/
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ashwin.V
RegisterNumber:212225040034
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([35, 50, 65, 70, 85])

model = LinearRegression()

model.fit(X, Y)

m = model.coef_[0]
b = model.intercept_

print("Slope (m):", m)
print("Intercept (b):", b)

x_input = float(input("Enter hours studied: "))
predicted_marks = model.predict([[x_input]])
print("Predicted Marks:", predicted_marks[0])

Y_pred = model.predict(X)

plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_pred, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression (Using sklearn)")
plt.legend()
plt.show()
```

## Output:
<img width="803" height="678" alt="image" src="https://github.com/user-attachments/assets/478e589f-6c82-44d6-b0f7-b0aabec64c01" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
