import os
import pandas as pd
import json  # Make sure to import json
import pyarrow
import zipfile
import pytest
import sys
from collections import defaultdict
from pycricketpred.data_wrangling import *
import helpers_data_wrangling as hdw

# test that a dataframe is returned
def test_parse_cricket_json():
    assert type(hdw.jsontest) == pd.DataFrame, "Dataframe is not returned"
    for col in hdw.cols:
        assert col in hdw.jsontest.columns, f"{col} is not included in the DataFrame"

# check that the parquet files are saved with >0 files
## this saving successfully also ensures that the parse_cricket_json runs successfully
def test_parquet_files_exist():
    assert os.path.isfile('tests/data/test_parquet/211028.parquet'), "Parquet file not saved"
    assert os.path.isfile('tests/data/test_parquet/211048.parquet'), "Parquet file not saved"
    assert os.path.isfile('tests/data/test_parquet/222678.parquet'), "Parquet file not saved"

# a zipped folder with no json files creates an empty folder
def test_empty_zipped_folder():
     assert os.path.isdir('tests/data/test_parquet_empty'), "Does not create empty parquet folder"

# check that the correct columns are added for the add_columns function
def test_check_columns_added():
    assert 'team_over' in hdw.data1.columns, "Team over not in dataframe"
    assert 'over_ball' in hdw.data1.columns, "Over_ball not in dataframe"
    assert 'inning' in hdw.data1.columns, "Inning not in dataframe"
    assert 'runs_cumulative' in hdw.data1.columns, "Runs_cumulative not in dataframe"
    assert 'powerplay' in hdw.data1.columns, "Powerplay not in dataframe"

def test_key_val_error():
    with pytest.raises(KeyError):
        add_columns(hdw.data_missing_cols)

# test majority dtypes returns the right objects
def test_majority_dtypes():
    assert type(hdw.majority1) == dict, "Did not return a dictionary"
    assert type(hdw.majority2) == dict, "Did not return a dictionary"

# check majority dtypes returns correct values
def test_majority_output():
    assert len(hdw.majority1) == hdw.jsontest.shape[1]

# check that concatenated dataframe function returns dataframe with right columns
def test_concat_output():
    assert type(hdw.concat1) == pd.DataFrame, "Does not return a dataframe"
    assert type(hdw.concat2) == pd.DataFrame, "Does not return a dataframe"
    for col in hdw.concat1.columns:
        if col == 'season':
            assert hdw.concat1[col].dtype == object, "Season is not string"
        else:
            assert hdw.concat1[col].dtype == hdw.majority1[col], f"Wrong dtype for column {col}"

## check that exceptions are raised
# check that an exception is raised when the parquet file is not processed
def test_parquet_exception():
    with pytest.raises(Exception):
        process_cricket_jsons('tests/data/test_zips_incorrect.zip', 'tests/data/outputs_incorrect')

# check that an exception is raised when the parquet file is not processed
def test_parquet_exception():
    with pytest.raises(Exception):
        determine_majority_dtypes(['230123.parquet'], 'tests/data/test_parquet_incorrect')

# check that an exception is raised when the parquet file is not processed
def test_parquet_exception():
    with pytest.raises(Exception):
        apply_dtypes_and_concatenate(['230123.csv'], 'tests/data/test_parquet_incorrect', hdw.majority1)
