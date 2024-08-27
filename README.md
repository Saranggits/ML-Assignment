- Explanation of Sarang ML Assignment 1:

- Step 1.Import Necessary Libraries: In python import pandas as pd from sklearn.datasets import load_iris

- Purpose: Import the pandas library, which is essential for data manipulation and analysis, and load_iris from sklearn.datasets to access the Iris dataset.

- Details: pandas is used to handle data in DataFrame format, which simplifies various data operations, while load_iris provides a convenient way to load the Iris dataset.

- Step 2. Load the Iris Dataset: python iris = load_iris()

- Purpose: Load the Iris dataset into the variable iris.

- Details: The load_iris function returns a dictionary-like object containing the data, feature names, and other metadata about the Iris dataset.

- Step 3.Convert to DataFrame: python iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

- Purpose: Convert the data from the Iris dataset into a pandas DataFrame named iris_df.

- Details: iris.data contains the feature data, and iris.feature_names provides the column names. This conversion makes it easier to manipulate and analyze the data using DataFrame methods.

- Step 4. Display the First Five Rows: python print("First five rows of the dataset:") print(iris_df.head())

- Purpose: Display the first five rows of the DataFrame to get a preview of the dataset.
Details: head() is a DataFrame method that returns the first five rows. This helps in quickly understanding the structure and content of the dataset.

- Step 5.Display the Shape of the Dataset: python print("\nShape of the dataset:") print(iris_df.shape)

- Purpose: Show the dimensions (number of rows and columns) of the DataFrame.
- Details: shape is an attribute of the DataFrame that returns a tuple representing the number of rows and columns. This provides a quick overview of the dataset size.

- Step 6.Display Summary Statistics: python print("\nSummary statistics of the dataset:") print(iris_df.describe())

- Purpose: Provide summary statistics for each feature in the DataFrame.

- Details: describe() is a DataFrame method that computes summary statistics such as mean, standard deviation, minimum, and maximum values for numerical columns.This helps in understanding the distribution and range of feature values.

- Explanation of Sarang ML Assignment 2:

- Step 1: Import Necessary Libraries In python from sklearn.datasets import load_iris from sklearn.model_selection import train_test_split

- from sklearn.datasets import load_iris*: This imports the load_irisfunction from thedatasetsmodule of thesklearnlibrary. Theload_iris` function is used to load the Iris dataset, which is a classic dataset in machine learning used for classification tasks.

- from sklearn.model_selection import train_test_split*: This imports the train_test_splitfunction from themodel_selectionmodule ofsklearn`. This function is used to split a dataset into training and testing sets. Step 2: Load the Iris Dataset IN python iris = load_iris() X = iris.data # Features y = iris.target # Target labels

- iris = load_iris()*: This line calls the load_iris` function, which returns a dataset object containing the Iris dataset. This object includes both the feature data and target labels.

- X = iris.data*: The dataattribute of theirisobject contains the feature data for each sample in the dataset. Here,X` is assigned to this feature data, which is a 2D array where each row represents a sample and each column represents a feature (e.g., sepal length, sepal width, petal length, petal width).

- y = iris.target*: The targetattribute of theirisobject contains the target labels for each sample, indicating the class of the iris plant (e.g., setosa, versicolor, or virginica). Here,yis assigned to these target labels, which is a 1D array where each element corresponds to the class of the respective sample inX`.

- Step 3: Split the Dataset into Training and Testing Sets In python X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

- train_test_split(X, y, test_size=0.2, random_state=42)`*: This function call splits the dataset into training and testing sets.

- Xandy: These are the feature data and target labels, respectively.

- test_size=0.2: This parameter specifies that 20% of the data should be allocated to the test set, while the remaining 80% will be used for training.

- random_state=42: This is a seed for the random number generator used to shuffle the data before splitting. Setting a random state ensures that the split is reproducible, meaning that you’ll get the same training and testing sets if you run the code again.

- The function returns four arrays: X_train, X_test, y_train, and y_test. These arrays represent the feature data and target labels for the training and testing sets, respectively.

- Step 4: Print the Number of Samples in Both Training and Testing Sets

- In python print(f"Number of training samples: {len(X_train)}") print(f"Number of testing samples: {len(X_test)}")

- len(X_train): This calculates the number of samples in the training set by finding the length of the X_train array.

- len(X_test): This calculates the number of samples in the testing set by finding the length of the X_test array

- The print statements output the number of samples in the training and testing sets to the console. This information helps verify that the split was done correctly and gives insight into the size of each dataset.

- Explanation of Sarang ML Assginment 3:

- Step 1. *Import necessary libraries:

- pandas: For data manipulation and creation of DataFrames.
- numpy: For numerical operations (though not strictly necessary here, it's often used).
- train_test_split from sklearn.model_selection: To split the dataset into training and testing sets.
- LinearRegression from sklearn.linear_model: To create the linear regression model.
- mean_squared_error from sklearn.metrics: To evaluate the model's performance.

- Step 2. Create a sample dataset:

- A dictionary data is defined with YearsExperience and Salary.
The dictionary is then converted into a DataFrame using pd.DataFrame().

- Step 3. Define features and target variable:

- X represents the feature (independent variable) which is YearsExperience.
- y represents the target variable (dependent variable) which is Salary.

- Step 4. Split the data into training and testing sets:

- The train_test_split function is used to split the data into training (80%) and testing (20%) sets.
random_state=42 ensures that the split is reproducible.

- Step 5. Create and fit the linear regression model:

- An instance of LinearRegression is created.
The model is fitted using the training data with the fit method.

- Step 6. Make predictions on the test set:

- The predict method is used to make predictions on the test data (X_test).

- Step 7. Calculate Mean Squared Error:
- mean_squared_error computes the MSE between the actual values (y_test) and the predicted values (y_pred).

- Step 8. Print the results:
- Output the MSE to evaluate the model’s performance.
