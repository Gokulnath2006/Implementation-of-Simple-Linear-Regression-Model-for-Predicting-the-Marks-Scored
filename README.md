# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import Libraries – Load Python libraries for data handling, visualization, and machine learning.
2.Load Data – Read the CSV file into a DataFrame and inspect the first and last few rows.
3.Split Features and Target – Separate the dataset into input features (X) and output target (y).
4.Split Data into Training and Testing Sets – Divide data into training and test sets for model evaluation.
5.Train Linear Regression Model – Fit a Linear Regression model using the training data.
6.Make Predictions – Predict the target values for the test data using the trained model.
7.Visualize Results – Plot actual vs predicted values for training and test sets to see the fit.
8.Evaluate Model – Calculate metrics like MSE, MAE, and RMSE to measure model accuracy.
 
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Gokul Nath R
RegisterNumber:  212224230077

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df= pd.read_csv('student_scores.csv')

df.head()
df.tail()

X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
y_pred

y_test

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

Head Values

<img width="503" height="270" alt="image" src="https://github.com/user-attachments/assets/e5249cd7-a270-4c7f-93c2-f47f9ce0891a" />


Tail Values

<img width="509" height="264" alt="image" src="https://github.com/user-attachments/assets/7cba426a-07ae-4cd1-915d-cc0e818e8410" />

X Values

<img width="519" height="613" alt="image" src="https://github.com/user-attachments/assets/b8cacd96-f652-434e-8b8e-b4430bad45f6" />

Y Values

<img width="761" height="124" alt="image" src="https://github.com/user-attachments/assets/4c17c8f3-65a8-4730-8c96-249310ab0a62" />

Predicted Values

<img width="848" height="282" alt="image" src="https://github.com/user-attachments/assets/9dd90766-3008-491d-891c-6c98f84a9142" />

Actual Values

<img width="757" height="89" alt="image" src="https://github.com/user-attachments/assets/0ba273bc-a8c6-4a73-a6e3-126b4420e78e" />

Training set

<img width="733" height="751" alt="image" src="https://github.com/user-attachments/assets/c477011d-f570-472b-86e5-ad59de750d4e" />

Testing set

<img width="747" height="739" alt="image" src="https://github.com/user-attachments/assets/538b5992-eb34-4fe9-b828-6615e18221c9" />

MSE, MAE and RMSE

<img width="547" height="227" alt="image" src="https://github.com/user-attachments/assets/2896bf5f-f478-4905-bae1-330310568971" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
