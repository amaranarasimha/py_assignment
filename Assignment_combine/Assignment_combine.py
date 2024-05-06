import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import math
import os

class IdealFunctionSelector:
    """
    A class to select ideal functions based on provided training and ideal datasets.

    Attributes:
    - train_file (str): Path to the training dataset CSV file.
    - test_file (str): Path to the test dataset CSV file.
    - ideal_file (str): Path to the ideal dataset CSV file.
    """

    def __init__(self, train_file, test_file, ideal_file):
        """
        Initialize IdealFunctionSelector with file paths.

        Args:
        - train_file (str): Path to the training dataset CSV file.
        - test_file (str): Path to the test dataset CSV file.
        - ideal_file (str): Path to the ideal dataset CSV file.
        """
        self.train_file = train_file
        self.test_file = test_file
        self.ideal_file = ideal_file

    def load_data(self, file_path):
        """
        Load data from CSV file.

        Args:
        - file_path (str): Path to the CSV file.

        Returns:
        - pandas.DataFrame: Loaded data.
        """
        try:
            data = pd.read_csv(file_path)
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None

    def fit_linear_regression(self, input_data, target_data):
        """
        Fit a linear regression model to input and target data.

        Args:
        - input_data (numpy.ndarray): Input data.
        - target_data (numpy.ndarray): Target data.

        Returns:
        - sklearn.linear_model.Ridge: Fitted linear regression model.
        """
        ridge = Ridge(alpha=1)
        ridge.fit(input_data.reshape(-1, 1), target_data)
        return ridge

    def choose_ideal_function(self, input_data, target_data, ideal_functions):
        """
        Choose the ideal function that minimizes the mean squared error.

        Args:
        - input_data (numpy.ndarray): Input data.
        - target_data (numpy.ndarray): Target data.
        - ideal_functions (list): List of pre-fitted ideal functions.

        Returns:
        - sklearn.linear_model.Ridge: Chosen ideal function.
        """
        min_error = math.inf
        chosen_function = None

        for func in ideal_functions:
            y_pred = func.predict(input_data.reshape(-1, 1))
            error = mean_squared_error(target_data, y_pred)

            if error < min_error:
                min_error = error
                chosen_function = func

        return chosen_function

    def map_test_data(self, input_test, target_test, ideal_functions):
        """
        Map test data to the closest ideal function.

        Args:
        - input_test (numpy.ndarray): Input test data.
        - target_test (numpy.ndarray): Target test data.
        - ideal_functions (list): List of pre-fitted ideal functions.

        Returns:
        - list: List of indices indicating the mapped ideal function for each test data point.
        """
        mapped_functions = []

        for i, (x, y) in enumerate(zip(input_test, target_test)):
            min_deviation = math.inf
            mapped_function = -1

            for j, func in enumerate(ideal_functions):
                deviation = abs(y - func.predict([[x]]))
                if deviation < min_deviation:
                    min_deviation = deviation
                    mapped_function = j

            mapped_functions.append(mapped_function)

        return mapped_functions

    def main(self):
        """
        Main method to execute the IdealFunctionSelector workflow.
        """
        # Load data
        train_df = self.load_data(self.train_file)
        test_df = self.load_data(self.test_file)
        ideal_df = self.load_data(self.ideal_file)

        if train_df is None or test_df is None or ideal_df is None:
            return

        # Define the features and target for training data
        X_train = train_df[['x']]
        y_train = train_df['y1']

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit the ridge regression model
        ridge = self.fit_linear_regression(X_train_scaled, y_train)

        # Define the features and target for test data
        X_test = test_df[['x']]
        y_test = test_df['y']
        X_test_scaled = scaler.transform(X_test)

        # Predict the target values for the test set
        y_pred = ridge.predict(X_test_scaled)

        # Plot the results
        plt.figure(figsize=(15, 6))
        plt.plot(train_df['x'], train_df['y1'], label='Training Data')
        plt.plot(test_df['x'], y_pred, color='red', label='Predicted Test Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Ridge Regression Prediction')
        plt.legend()
        plt.grid(True)
        plt.show()

        return train_df, test_df, ideal_df


if __name__ == "__main__":
    # Initialize IdealFunctionSelector with file paths
    selector = IdealFunctionSelector(
        'C:\\Users\\AmarReddy\\Downloads\\dataset\\train.csv',
        'C:\\Users\\AmarReddy\\Downloads\\dataset\\test.csv',
        'C:\\Users\\AmarReddy\\Downloads\\dataset\\ideal.csv'
    )
    # Execute the main method
    train_df, test_df, ideal_df = selector.main()

    print("Train DataFrame:")
    print(train_df.head())
    print("\nTest DataFrame:")
    print(test_df.head())
    print("\nIdeal DataFrame:")
    print(ideal_df.head())
