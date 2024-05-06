# # -*- coding: utf-8 -*-
# """
# Created on Sat Nov  4 21:01:59 2023

# @author: AmarReddy
# """

import pandas as pd
# For plotting with bokeh
from bokeh.plotting import figure, show
from bokeh.models import Legend, LegendItem
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import math


# Load the data
train_df = pd.read_csv('C:\\Users\\AmarReddy\\Downloads\\dataset\\train.csv')
test_df = pd.read_csv('C:\\Users\\AmarReddy\\Downloads\\dataset\\test.csv')
ideal_df = pd.read_csv('C:\\Users\\AmarReddy\\Downloads\\dataset\\ideal.csv')


print(train_df.head())

lr = LinearRegression()
input= train_df[:]['x']
input= input.values.reshape(-1,1)
# exit()
y1= train_df[:]['y1']
y1= y1.values.reshape(-1,1)
y2= train_df[:]['y2']
y2= y2.values.reshape(-1,1)
y3= train_df[:]['y3']
y3= y3.values.reshape(-1,1)
y4= train_df[:]['y4']
y4= y4.values.reshape(-1,1)

func1= lr.fit(input, y1)
func2= lr.fit(input, y2)
func3= lr.fit(input, y3)
func4= lr.fit(input, y4)


ideal_functions= []

def calculate_ideal_fn():
    x= ideal_df[:]['x']
    x= x.values.reshape(-1,1)
    for i in range (50):
        y= ideal_df[:]['y'+ str(i+1)]
        y= y.values.reshape(-1,1)

        func= lr.fit(x,y)
        ideal_functions.append(func)


calculate_ideal_fn()
print(len(ideal_functions))
print("functions approximated correctly")


def choose_ideal_fn(x,y):
    index= 0
    min_error= 1000
    for i in range(50):
        func= ideal_functions[i]
        y_pred= func.predict(x)
        error= mean_squared_error(y_pred, y)

        if(error < min_error):
            index= i
            min_error= error

    return ideal_functions[index]

choosen_ideal_fn1= choose_ideal_fn(input, y1)
choosen_ideal_fn2= choose_ideal_fn(input, y2)
choosen_ideal_fn3= choose_ideal_fn(input, y3)
choosen_ideal_fn4= choose_ideal_fn(input, y4)

choosen_ideal_fn= [choosen_ideal_fn1, choosen_ideal_fn2, choosen_ideal_fn3, choosen_ideal_fn4]

print("ideal functions are choosen correctly")

# exit()
input_test= test_df[:]['x']
input_test= input_test.values.reshape(-1,1)
# exit()
y_test= test_df[:]['y']
y_test= y_test.values.reshape(-1,1)


mapped_ideal_fn= []
def mapping_ideal_for_test(input_test, y_test, input, y1,y2,y3,y4, choosen_ideal_fn, mapped_ideal_fn):
    max_dev1 = math.sqrt(max(abs(y1 - choosen_ideal_fn[0].predict(input))))
    max_dev2 = math.sqrt(max(abs(y2 - choosen_ideal_fn[1].predict(input))))
    max_dev3 = math.sqrt(max(abs(y3 - choosen_ideal_fn[2].predict(input))))
    max_dev4 = math.sqrt(max(abs(y4 - choosen_ideal_fn[3].predict(input))))

    for pair in zip(input_test,y_test):
        xx= pair[0].reshape(-1,1)
        # exit()

        dev1 = (abs(pair[1] - choosen_ideal_fn1.predict(xx)))
        dev2 = (abs(pair[1] - choosen_ideal_fn2.predict(xx)))
        dev3 = (abs(pair[1] - choosen_ideal_fn3.predict(xx)))
        dev4 = (abs(pair[1] - choosen_ideal_fn4.predict(xx)))


        if dev1 < max_dev1:
            mapped_ideal_fn.append(0)
        elif dev2 < max_dev2:
            mapped_ideal_fn.append(1)
        elif dev3 < max_dev3:
            mapped_ideal_fn.append(2)
        elif dev4 < max_dev4:
            mapped_ideal_fn.append(3)
        else:
            mapped_ideal_fn.append(-1)
            

mapping_ideal_for_test(input_test, y_test, input, y1,y2,y3,y4, choosen_ideal_fn, mapped_ideal_fn)
print(len(mapped_ideal_fn))




