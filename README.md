# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import essential libraries like pandas, numpy, and sklearn. Load the dataset containing employee features (e.g., experience, education) and the target (salary).
2.Handle missing values, encode categorical features (e.g., department) using OneHotEncoder or LabelEncoder. Normalize or scale numerical features if necessary to ensure consistent data for the model.
3.Use train_test_split() from sklearn.model_selection to divide the data into training and testing sets. Typically, you split the data into 80% for training and 20% for testing.
4.Use DecisionTreeRegressor from sklearn.tree and fit the model on the training data. You can specify parameters such as max_depth or min_samples_split to control the complexity of the tree.
5.Use the trained model to predict the salary for the employees in the test dataset. Store these predictions and compare them with the actual salaries for evaluation.
6.Use regression evaluation metrics like mean_squared_error, r2_score, or mean_absolute_error from sklearn.metrics. These metrics will help assess how well the model predicts employee salaries.
7.Use plot_tree() from sklearn.tree to visualize the structure of the decision tree. This will give insights into the most important features influencing salary predictions and how the tree splits.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MITHUN G
RegisterNumber:  212223080030
*/
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull.sum()
data["Salary"].value_counts()
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2 = metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:

Dataset

![image](https://github.com/user-attachments/assets/aeb6e155-d8ca-41a0-8602-ebcabbabc95e)

Data Info

![image](https://github.com/user-attachments/assets/dc9115f6-add4-473e-bb1d-a28fcbdb333c)

Sum of Null Values

![image](https://github.com/user-attachments/assets/b3548bf4-98ad-45b7-80a9-40a7e8158606)


Value count of Salary column in data set


![image](https://github.com/user-attachments/assets/8679a37a-810a-4263-b765-bd303e816faa)


Labelling Position column

![image](https://github.com/user-attachments/assets/21fa7f29-6ede-4d03-aca1-07928ab7d14c)


Assigning x and y values

![image](https://github.com/user-attachments/assets/19565b1c-efe9-4c6a-8b37-be67516b5c0a)
![image](https://github.com/user-attachments/assets/d1bd999c-77c3-4914-9dce-8fe27fd13427)


Mean Squared Error

![image](https://github.com/user-attachments/assets/4a4d8a0a-cea4-49d0-b5c0-25e185ab77b4)

R2

![image](https://github.com/user-attachments/assets/62ba86ce-6788-470c-a9f0-ea6e4e73357e)

Prediction

![image](https://github.com/user-attachments/assets/0dc08484-7228-45b0-89c0-b6bd7a6c48fa)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
