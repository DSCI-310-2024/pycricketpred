import numpy as np
import pandas as pd
import altair as alt
import os
import sys
import matplotlib.pyplot as plt
from pycricketpred.eda import *

# example inputs

## testing for dataframe + column validity
data1 = pd.DataFrame({
    'A': [2, 3, 5],
    'B': [0,1,1], 
    'wicket':[1, 0, 1]
})

data2 = pd.Series({})
data3 = pd.DataFrame({
    'A': ["hello", "goodbye"],
    'B': ["b", "a"]
})
data4 = pd.DataFrame({})

x_input1 = 'A'
x_input2 = 'a'
x_input3 = 10
x_input4 = 'B:N'
x_input5 = 'F:N'

## testing for figure input validity
width1 = 10
height1 = 20
width2 = "a"
height2 = 20.5

## testing for file path validity
chart1 = "test_chart1.png"
chart2 = 1

filepath1 = "tests/images/"
filepath2 = 20

## testing for figure output validity
result1 = vis_bar(data1, x_input1, width1, height1)
result2 = hist_chart(data1, x_input1, chart1, filepath1)
result3 = vis_bar(data1, x_input4, width1, height1)