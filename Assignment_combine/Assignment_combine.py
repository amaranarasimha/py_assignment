import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import math
import os

class IdealFunctionSelector:
    def __init__(self, train_file, test_file, ideal_file):
        self.train_file = train_file
        self.test_file = test_file
        self.ideal_file = ideal_file

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error loading data from {file_path}: {str(e)}")
            return None

    def fit_linear_regression(self, input_data, target_data):
        ridge = Ridge(alpha=1)
        ridge.fit(input_data, target_data)
        return ridge

    def choose_ideal_function(self, input_data, target_data, ideal_functions):
        min_error = math.inf
        chosen_function = None

        for func in ideal_functions:
            y_pred = func.predict(input_data)
            error = mean_squared_error(target_data, y_pred)

            if error < min_error:
                min_error = error
                chosen_function = func

        return chosen_function

    def map_test_data(self, input_test, target_test, ideal_functions):
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
        train_df = self.load_data(self.train_file)
        test_df = self.load_data(self.test_file)
        ideal_df = self.load_data(self.ideal_file)

        if train_df is None or test_df is None or ideal_df is None:
            return None, None, None

        X_train = train_df[['x']]
        y_train = train_df['y1']
        X_test = test_df[['x']]
        y_test = test_df['y']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge = self.fit_linear_regression(X_train_scaled, y_train)

        y_pred = ridge.predict(X_test_scaled)

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
    # Set the paths to the dataset files
    train_file = "C:\\Users\\AmarReddy\\Downloads\\dataset\\train.csv"
    test_file = "C:\\Users\\AmarReddy\\Downloads\\dataset\\test.csv"
    ideal_file = "C:\\Users\\AmarReddy\\Downloads\\dataset\\ideal.csv"

    selector = IdealFunctionSelector(train_file, test_file, ideal_file)
    train_df, test_df, ideal_df = selector.main()

    print("Train DataFrame:")
    print(train_df.head())
    print("\nTest DataFrame:")
    print(test_df.head())
    print("\nIdeal DataFrame:")
    print(ideal_df.head())
