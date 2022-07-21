# Stephen FitzSimon
# possum regression project model_target_sex.py files
# Contains functions to prepare the possum dataframe for modelling of 
# the sex target variable

import pandas as pd
import sklearn.neighbors as neigh

# global constants to help with modeling

# ensures that the models are always the same as in the notebook
RAND_SEED = 1729 

# holds all of the names of measurement columns
MEASUREMENT_COLUMNS = [
    'total_length',
    'body_length',
    'tail_length',
    'head_length',
    'skull_width',
    'foot_length',
    'eye_width',
    'chest_girth',
    'belly_girth',
    'ear_length'
]

# holds all the names of the ratio columns
RATIO_COLUMNS = [
    'body_head_ratio',
    'body_eye_ratio', 
    'body_skull_ratio',
    'total_head_ratio',
    'total_eye_ratio',
    'total_skull_ratio',
    'belly_head_ratio',
    'belly_skull_ratio',
    'belly_eye_ratio',
    'chest_head_ratio',
    'chest_skull_ratio',
    'foot_eye_ratio'
]

def make_modeling_columns(df):
    """
    Makes the calculated columns used in modeling
    """
    df['body_length'] = df['total_length'] - (df['tail_length'] + df['head_length'])
    df['body_head_ratio']  = df['body_length']/df['head_length']
    df['body_eye_ratio'] = df['body_length']/df['eye_width']
    df['body_skull_ratio'] = df['body_length']/df['skull_width']
    df['total_head_ratio'] = df['total_length']/df['head_length']
    df['total_eye_ratio'] = df['total_length']/df['eye_width']
    df['total_skull_ratio'] = df['total_length']/df['skull_width']
    df['belly_head_ratio'] = df['belly_girth']/df['head_length']
    df['belly_skull_ratio'] = df['belly_girth']/df['skull_width']
    df['belly_eye_ratio'] = df['belly_girth']/df['eye_width']
    df['chest_head_ratio'] = df['chest_girth']/df['head_length']
    df['chest_skull_ratio'] = df['chest_girth']/df['skull_width']
    df['foot_eye_ratio'] = df['foot_length']/df['eye_width']
    return df

def make_X_and_y(df):
    """
    Makes a X and y sets
    """
    #drop relevant columns
    X_df = df.drop(columns = ['sex'])
    #make y_Train
    y_df = df[['case', 'sex']]
    return X_df, y_df