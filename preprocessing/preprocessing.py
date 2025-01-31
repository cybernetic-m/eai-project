import pandas as pd
import os
import torch
import numpy as np
from numpy.polynomial.polynomial import Polynomial as Poly


def normalize_columns(df: pd.DataFrame, columns: list, new_min: float, new_max: float) -> pd.DataFrame:
    """
    Normalizes the specified columns in the DataFrame between new_min and new_max.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    columns (list): A list of column names to normalize.
    new_min (float): The minimum value of the normalized range.
    new_max (float): The maximum value of the normalized range.

    Returns:
    pd.DataFrame: A new DataFrame with the specified columns normalized.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    for column in columns:
        if column in df_copy.columns:
            min_val = df_copy[column].min()
            max_val = df_copy[column].max()

            # Avoid division by zero if the column has a constant value
            if max_val != min_val:
                df_copy[column] = new_min + (df_copy[column] - min_val) * (new_max - new_min) / (max_val - min_val)
            else:
                # If the column has constant values, normalize to new_min (or new_max, depends on the case)
                df_copy[column] = new_min
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

    return df_copy


def merge_on_closest_time(df1, df2, time_col='time'):
    # Ensure the 'time' columns are datetime objects
    df1[time_col] = pd.to_datetime(df1[time_col])
    df2[time_col] = pd.to_datetime(df2[time_col])

    # Sort both dataframes by time (if not already sorted)
    df1 = df1.sort_values(by=time_col).reset_index(drop=True)
    df2 = df2.sort_values(by=time_col).reset_index(drop=True)

    # Create an empty list to hold the merged rows
    merged_rows = []

    # Initialize pointers for both DataFrames
    i, j = 0, 0

    # Iterate through each row in df1 and find the closest time in df2
    while i < len(df1):
        row1 = df1.iloc[i]

        # Move pointer j to the closest time in df2 (either before or after row1)
        while j < len(df2) - 1 and abs(df2.iloc[j + 1][time_col] - row1[time_col]) < abs(df2.iloc[j][time_col] - row1[time_col]):
            j += 1

        # Get the closest row from df2
        closest_row = df2.iloc[j]

        # Combine the row from df1 with the closest row from df2
        merged_row = pd.concat([row1, closest_row], axis=0)

        # Reset the index of the concatenated row before appending to avoid duplicate indices
        merged_rows.append(merged_row.reset_index(drop=True))

        # Move the pointer for df1
        i += 1

    # Create a new DataFrame from the merged rows and reset the index
    merged_df = pd.DataFrame(merged_rows)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df

def transform_dataframe(df):
    # Create an empty list to store the filled rows
    filled_rows = []

    # Iterate through each row in the DataFrame
    for i, row in df.iterrows():
        # If X1 is not NaN, we know this is a valid row for X1
        if pd.notna(row['X1']):
            x1_value = row['X1']
            filled_rows.append({'time': row['time'], 'X1': x1_value, 'Y1': df.iloc[i+1]['Y1'], 'Z1': df.iloc[i+2]['Z1']})

    # Convert the list of filled rows to a DataFrame
    filled_df = pd.DataFrame(filled_rows)

    # Now group the data by time and use the last available X1, Y1, and Z1 for each time
    final_df = filled_df.groupby('time').last().reset_index()

    return final_df


def load_dataset_dict_pandas(csv_path):

    # Initialization of the dictionary of input data (text) {'filename': ['hi ...','I am ...']}
    text_dict = {}

    filelist = os.listdir(csv_path) # List of all csv file ['filename_1.csv', ...]

    # Iterate over all csv file in the directory
    for filename in filelist:
        # Initialization of the list containing all the texts in a csv file

        text_dict[filename] = []
        # Open the i-th csv file

        test = pd.read_csv(csv_path + '/' + filename)
        print(filename)
        print(test.info())

def my_gradient(dataset, window_size):
    
    # Instantiate the three lines to compute the gradient (estimated as the angular coefficient of the lines)
    x_line = Poly(coef=[0,1])
    y_line = Poly(coef=[0,1])
    z_line = Poly(coef=[0,1])

    # Create the lists for all the gradients (len(dataset) - window_size gradients)
    gradients = []

    samples = [] # List of 100 temporary samples
    for i in range(window_size, len(dataset)):
        samples = dataset[i-window_size:i] # Take 100 elements of X1, Y1, Z1 
        
        # Take all rows of the first columns X1, Y1, Z1
        x1_data = samples[:,0] 
        y1_data = samples[:,1] 
        z1_data = samples[:,2] 
        
        # Fit a line to 100 points of X1, Y1, Z1
        x1_line = x_line.fit(x=range(window_size), y=x1_data, deg=1) 
        y1_line = y_line.fit(x=range(window_size), y=y1_data, deg=1) 
        z1_line = z_line.fit(x=range(window_size), y=z1_data, deg=1) 
        
        # Return the coefficients in form [q,m] and take the angular coefficients
        x1_m = x1_line.convert().coef[1] 
        y1_m = y1_line.convert().coef[1] 
        z1_m = z1_line.convert().coef[1]

        # Append the value of the gradients
        gradients.append([x1_m, y1_m, z1_m])

    return np.array(gradients) 

        