# pycricketpred
[![codecov](https://codecov.io/gh/DSCI-310-2024/pycricketpred/graph/badge.svg?token=vzmjekckdP)](https://codecov.io/gh/DSCI-310-2024/pycricketpred)

A package for predicting and find interesting cricket related statistics!

## Installation

```bash
$ pip install pycricketpred
```

## Usage

The `pycricketpred` package has 4 modules (`data_wrangling`, `data_cleaning`, `eda` and `modelling`) which perform a wide variety of functions on ball-by-ball cricket data, in the `json` format. 

```
from pycricketpred.data_wrangling import *
from pycricketpred.data_cleaning import *
from pycricketpred.eda import *
from pycricketpred.modelling import *

# parse through json files in a zipped archive and convert the data into parquet format

process_cricket_jsons('data/t20s_json', 'data/t20s_parquet')

# create majority dtype mapping dictionary and create a dataframe from the mapping

majority_mapping = determine_majority_dtypes(['2203.parquet', '21332.parquet'], 'data/t20s_parquet')
apply_dtypes_and_concatenate(['2203.parquet', '21332.parquet'], 'data/t20s_parquet', majority_mapping)

# create histograms for variable distribution, specifying width and height

vis_bar(cricket_df, 'over', 10, 20)

# Create and save histograms for wicket distribution across categories

hist_chart(cricket_df, 'over', 'chart1.png', 'images/')
```
and perform a wide range of modelling tasks, including splitting data, preprocessing, and creating and evaluating classification models using a confusion matrix.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pycricketpred` was created by DSCI 310 Group 11: Alex Lin, Jackson Siemens, Shruti Vijaykumar Seetharam, Hanlin Zhao. It is licensed under the terms of the MIT license.

## Credits

`pycricketpred` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
