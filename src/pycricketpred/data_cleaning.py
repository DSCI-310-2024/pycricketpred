import pandas as pd
import os
from sklearn.model_selection import train_test_split


def separate_columns(dataframe):
    """ Drop the 'wicket' column from the DataFrame and separate it as the target variable.

    Parameters
    ----------
    data: pd.DataFrame
        The input dataset that contains all the data

    Returns
    ----------
    X: pd.DataFrame
        DataFrame with the target column ('wicket') dropped.
        
    y: pd.Series
        Series containing the 'wicket' column (target variable).

    Raises
    ----------
        TypeError: If data is not a pandas DataFrame or if 'wicket' column is not present.
    
    Example
    ----------
    >>> separate_columns(cricket_df)
    """
    # Check if data is a pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    # Check if 'wicket' column is present
    if 'wicket' not in dataframe.columns:
        raise ValueError("DataFrame must contain 'wicket' column")

    # Drop the 'wicket' column from the DataFrame
    X  = dataframe.drop(columns=['wicket'])
    
    # Separate 'wicket' column as the target variable
    y = dataframe['wicket']

    return X, y



def split_and_save_data(X, y, train_size=0.7, save_table_path="."):
    """ Split the data into training and testing sets, and save the training data to a CSV file.

    Parameters
    ----------
    X: pd.DataFrame
        The dataframe with all the predictor variables

    y: pd.Series
        Series containing the target variable.

    train_size: float
        The proportion of the dataset to include in the training set.

    random_state: int or None
        Random seed for reproducibility.
        
    save_table_path: str
        The path to save the CSV file.

    Returns
    ----------
    None

    Raises
    ----------
    TypeError: If X is not a pandas DataFrame, y is not a pandas Series, or save_table_path is not a string.
    
    ValueError: If train_size is not between 0 and 1.
    
    IOError: If there is an issue saving the CSV file.

    Example
    ----------
    >>> from pycricketpred.data_cleaning import separate_columns
    >>> X, y = separate_columns(cricket_df)
    >>> split_and_save_data(X, y)
    """
    # Check if X is a pandas DataFrame
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    
    # Check if y is a pandas Series or DataFrame
    if not isinstance(y, (pd.DataFrame, pd.Series)):
        raise TypeError("y must be a pandas DataFrame or Series")
    
    # Check if train_size is between 0 and 1
    if not 0 < train_size < 1:
        raise ValueError("train_size must be a float between 0 and 1")

    # Check if save_table_path is a string
    if not isinstance(save_table_path, str):
        raise TypeError("save_table_path must be a string")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False,)
    
    # Concatenate X_train and y_train
    train_data = pd.concat([X_train, y_train], axis=1)

    # Create save path directory if it doesn't exist
    if not os.path.exists(save_table_path):
        os.makedirs(save_table_path)

    # Save the training data to a CSV file
    try:
        train_data.to_csv(os.path.join(save_table_path, "train_data.csv"), index=False)
    except IOError as e:
        # Handle IOError
        print("Error occurred while saving the CSV file:", e)
    
    return X_train, X_test, y_train, y_test, train_data
