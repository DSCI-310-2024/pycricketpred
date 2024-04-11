import numpy as np
import pandas as pd
import altair as alt
import os
import matplotlib.pyplot as plt


def vis_bar(data, x_input, width, height):
    """ Plot the distribution of a specified variable in the dataset

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe to plot the distribution based off of
    
    x_input: str
        Column to plot the distribution for
    
    width: int
        Width of the plot
    
    height: int
        Height of the plot
    
    Returns
    ----------
    alt.Chart
        A chart which shows the distribution of a column in a dataframe

    Examples 
    ----------
    >>> vis_bar(data_cricket, 'over', 10, 20)

    """

    # make sure width and height are correct input types
    if type(width) != int or type(height) != int:
        raise TypeError("Width and Height must be integers")
    
    # make sure column name is correct input type
    elif type(x_input) != str:
        raise TypeError("X input must be a string")
    
    # ensure dataframe is not empty
    elif data.empty == True:
        raise ValueError("DataFrame shouldn't be empty")
    
    # if column is nominal, we include :N at the end. ensure that the actual column name is in the dataframe
    elif x_input[-2:] == ':N' and x_input[:-2] not in data.columns:
        raise KeyError("Column must be in DataFrame")

    elif x_input[-2:] != ':N' and x_input not in data.columns:
        raise KeyError("Column must be in DataFrame")
    # create chart
    else:

        # transform nominal columns
        if x_input[-2:] == ':N':
            x_input = x_input[:-2]
            data[x_input] = data[x_input].astype(str)
        chart = alt.Chart(data).mark_bar().encode(
            x = x_input,
            y = "count()"
        ).properties(
            width = width, 
            height = height
        )
    return chart

    
def hist_chart(data, col, chart_name, save_path):
    """ Create and save distribution of wickets across different categories

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe based off of which the plots are created
    
    column: str
        Category for which the wicket distribution will be plotted. Has to be a categorical variable.

    chart_name: str
        File name to save the chart as

    save_path: str
        File path to save the chart to
    
    Returns 
    ----------
    None

    Examples 
    ----------
    >>> hist_chart(data_cricket, 'inning', 'chart1.png', 'images/')
    """
    # check for type of filepath/name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if type(chart_name) != str or type(save_path) != str:
        raise TypeError("Chart name and file paths must be strings")
    
    # check for type of column name
    elif type(col) != str:
        raise TypeError("X input must be a string")
    
    # ensure dataframe is not empty
    elif data.empty == True:
        raise ValueError("DataFrame shouldn't be empty")
    
    # ensure dataframe contains the necessary columns
    elif col not in data.columns or 'wicket' not in data.columns:
        raise KeyError("Column must be in DataFrame")

    else:
        count_wicket = data.groupby(col)['wicket'].sum()
        chart = count_wicket.plot(kind = 'bar', xlabel=f"{col}", ylabel="Wicket Count")
        fig = chart.get_figure()
        fig.savefig(os.path.join(save_path, chart_name))