import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print("First five rows of the Iris dataset:")
print(iris_df.head())
print("\nDataset shape:")
print(iris_df.shape)
print("\nSummary statistics:")
print(iris_df.describe())
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(iris_df, y, test_size=0.2, random_state=42)
print("\nNumber of samples in the training set:", X_train.shape[0])
print("Number of samples in the testing set:", X_test.shape[0])
np.random.seed(0)
years_experience = np.random.rand(100, 1) * 10  
salary = years_experience * 5000 + (np.random.rand(100, 1) * 10000) 
X_train_exp, X_test_exp, y_train_salary, y_test_salary = train_test_split(years_experience, salary, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train_exp, y_train_salary)
y_pred_salary = model.predict(X_test_exp)
mse = mean_squared_error(y_test_salary, y_pred_salary)
print("\nMean Squared Error (MSE) of the linear regression model:", mse)
