import numpy as np
import pandas as pd
import altair as alt
import os 
import pytest
import sys
import matplotlib.pyplot as plt
from pycricketpred.eda import *
import helpers_eda as hp_eda

# test for correct vis_bar outputs
def test_vis_bar_output():
    assert isinstance(hp_eda.result1, alt.Chart), "Output is not an Altair Chart"
    assert hp_eda.result1.encoding.x.shorthand == hp_eda.x_input1, "X-axis is not x-axis input"
    assert hp_eda.result3.encoding.x.shorthand + ":N" == hp_eda.x_input4, "X-axis is not x-axis input"
    assert hp_eda.result1.encoding.y.shorthand == 'count()', "X-axis is not x-axis input"
    assert hp_eda.result1.width == hp_eda.width1, "Width is not width specified"
    assert hp_eda.result1.height == hp_eda.height1, "Height is not height specified"

# test for correct histogram outputs
def test_hist_output():
    assert os.path.isfile("tests/images/test_chart1.png"), "Image is not saved"

# test key error for vis_bar function
def test_column_error():
    with pytest.raises(KeyError):
        vis_bar(hp_eda.data1, hp_eda.x_input2, hp_eda.width1, hp_eda.height1)

# test key error for hist_chart function
def test_wicket_error():
    with pytest.raises(KeyError):
        hist_chart(hp_eda.data3, hp_eda.x_input1, hp_eda.chart1, hp_eda.filepath1)

# test that a key error is created for nominal columns not in the dataframe
def test_nominal_key_error():
    with pytest.raises(KeyError):
        vis_bar(hp_eda.data1, hp_eda.x_input5, hp_eda.width1, hp_eda.height1)

# test type error for width/height
def test_int_error():
    with pytest.raises(TypeError):       
        vis_bar(hp_eda.data1, hp_eda.x_input1, hp_eda.width2, hp_eda.height2)

# test type error is raised for non-string input for column name
def test_str_error():
    with pytest.raises(TypeError):
        vis_bar(hp_eda.data1, hp_eda.x_input3, hp_eda.width1, hp_eda.height1)

# test that a type error is raised for non-string input for column name
def test_his_col_error():
    with pytest.raises(TypeError):
        hist_chart(hp_eda.data1, hp_eda.x_input3, hp_eda.chart1, hp_eda.filepath1)

# test that a type error is raised for incorrect filepath format
def test_filepath_error():
    with pytest.raises(TypeError):
        hist_chart(hp_eda.data1, hp_eda.x_input1, hp_eda.chart1, hp_eda.filepath2)

# test that a type error is raised for incorrect chart name format
def test_chart_name_error():
    with pytest.raises(TypeError):
        hist_chart(hp_eda.data1, hp_eda.x_input1, hp_eda.chart2, hp_eda.filepath1)
        
# test value error for empty dataframe
def test_value_error():
    with pytest.raises(ValueError):
        vis_bar(hp_eda.data4, hp_eda.x_input1, hp_eda.width1, hp_eda.height1)


