import os
import pandas as pd
import json  
import pyarrow
import zipfile
import sys
from collections import defaultdict
from pycricketpred.data_wrangling import *

# sample dataframes
data = pd.DataFrame({
    'over': [1, 2, 3],
    'team': ["England", "New Zealand", "England"],
    'runs_total': [100, 52, 31]
})

data_missing_cols = pd.DataFrame({
    'A': ["hello", "goodbye"],
    'B': ["b", "a"]
})

data1 = add_columns(data)

# list to check columns
cols = ["game_id","season","team","over", "batter","batter_id","bowler","bowler_id","non_striker","non_striker_id",
        "wides", "noballs","legbyes","byes","wicket","player_out","player_out_id","fielders_name","fielders_id",
        "wicket_type","runs_batter","runs_extras","runs_total"]

# sample json file
with open('tests/data/211028.json', 'r') as file:
    jsontest = parse_cricket_json(file, '211028')
    jsontest = add_columns(jsontest)

# sample zipped folder
process_cricket_jsons('tests/data/test_zips.zip', 'tests/data/test_parquet')
process_cricket_jsons('tests/data/test_zip_empty.zip', 'tests/data/test_parquet_empty')

# sample to determine majority types
majority1 = determine_majority_dtypes(['211028.parquet', '211048.parquet', '222678.parquet'],
                                       'tests/data/test_parquet')
majority2 = determine_majority_dtypes([], 'tests/data/test_parquet_empty')

# sample concatenated output
concat1 = apply_dtypes_and_concatenate(['211028.parquet', '211048.parquet', '222678.parquet'],
                                       'tests/data/test_parquet', majority1)
concat2 = apply_dtypes_and_concatenate([], 'tests/data/test_parquet_empty', majority1)