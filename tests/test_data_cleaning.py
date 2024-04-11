import pandas as pd
import sys
import os
import pytest
from sklearn.model_selection import train_test_split
from pycricketpred.data_cleaning import *
import helpers_data_cleaning as hp_dc


# Fixture to provide test DataFrame
@pytest.fixture
def dataframe():
    data = hp_dc.df_fake_test
    return data


# Test the functionality of the test_separate_columns function.
def test_separate_columns_function(dataframe):

    # Call the function being tested
    X, y = separate_columns(dataframe)

    # Check for the correct data types
    assert isinstance(X, pd.DataFrame), "Error: X should be a pandas DataFrame."
    assert isinstance(y, pd.Series), "Error: y should be a pandas Series."
    # Assert statements to check the functionality
    assert 'wicket' not in X.columns, "Error: 'wicket' column should be dropped from X."
    assert 'wicket' in y.name, "Error: 'wicket' column should be included in y."
    assert len(X) == len(y), "Error: Length of X and y should match."


# Test function for train_test_split_and_concat
def test_train_test_split_and_concat():
    
    # Perform train test split and concatenate
    X_train, X_test, y_train, y_test,train_data = split_and_save_data(
        hp_dc.X_fake, hp_dc.y_fake, train_size=0.7, save_table_path="tests/data/")


    # Assertions
    # Assert that the shapes of X_train, X_test, y_train, and y_test are correct
    assert X_train.shape[0] == int(0.7 * hp_dc.X_fake.shape[0]), "Incorrect number of samples in X_train"
    assert X_test.shape[0] == hp_dc.X_fake.shape[0] - X_train.shape[0], "Incorrect number of samples in X_test"
    assert y_train.shape[0] == int(0.7 * hp_dc.y_fake.shape[0]), "Incorrect number of samples in y_train"
    assert y_test.shape[0] == hp_dc.y_fake.shape[0] - y_train.shape[0], "Incorrect number of samples in y_test"

    # Assert that train_data has the correct number of rows and columns
    assert train_data.shape[0] == X_train.shape[0], "Incorrect number of rows in train_data"
    assert train_data.shape[1] == X_train.shape[1] + 1, "Incorrect number of columns in train_data"

    # Assert that y column "wicket" is present in train_data
    assert 'wicket' in train_data.columns, "'wicket' column is missing in train_data"

    # Assert that X_train and X_train_fake are the same
    assert X_train.equals(hp_dc.X_train_fake), "X_train and X_train_fake are not the same"

    # Assert that X_test and X_test_fake are the same
    assert X_test.equals(hp_dc.X_test_fake), "X_test and X_test_fake are not the same"

    # Assert that y_train and y_train_fake are the same
    assert y_train.equals(hp_dc.y_train_fake), "y_train and y_train_fake are not the same"

    # Assert that y_test and y_test_fake are the same
    assert y_test.equals(hp_dc.y_test_fake), "y_test and y_test_fake are not the same"

    # Assert that the data is saved
    assert os.path.isfile('tests/data/test_parquet/train_data.csv'), "Train data is not saved"

# Test that a type error is raised if the dataframe is empty
def test_df_type_error():
    with pytest.raises(TypeError):
        separate_columns(hp_dc.df_empty), "Dataframe is empty"

# Test that a value error is raised if the wicket column is not in the df
def test_wicket_value_error():
    with pytest.raises(ValueError):
        separate_columns(hp_dc.data_x), "Wicket column is not in the dataframe"

# Test that a type error is raised if X is not a dataframe
def test_X_type_error():
    with pytest.raises(TypeError):
        split_and_save_data(hp_dc.dict_X, hp_dc.y_fake), "X is not a dataframe"

# Test that a type error is raised if y is not a dataframe or series
def test_y_type_error():
    with pytest.raises(TypeError):
        split_and_save_data(hp_dc.X_fake, hp_dc.data_y), "y is not a dataframe or series"

# Test that a value error is raised if the train_size passed is less than 0
def test_train_size_error_less():
    with pytest.raises(ValueError):
        split_and_save_data(hp_dc.X_fake, hp_dc.y_fake, train_size=-100), "train_size is not between 0 and 1"

# Test that a value error is raised if the train_size passed is greater than 1
def test_train_size_error_greater():
    with pytest.raises(ValueError):
        split_and_save_data(hp_dc.X_fake, hp_dc.y_fake, train_size=20), "train_size is not between 0 and 1"

# Test that a type error is raised if the file_path given is not a string
def test_file_path_error():
    with pytest.raises(TypeError):
        split_and_save_data(hp_dc.X_fake, hp_dc.y_fake, save_table_path=23), "File path given is not a string"

# Test that a filepath is created when none exists
def test_file_path_creation():
    assert os.path.isfile('tests/data/fake_path/train_data.csv'), "Directory not created"

# Test that an IO error is raised
def test_io_error():
    with pytest.raises(IOError):
        split_and_save_data(hp_dc.X_fake, hp_dc.y_fake, save_table_path='23.'), "IOError not raised"