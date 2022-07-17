# Stephen FitzSimon
# possum regression project univariate_exploration.py files
# Contains functions to visualize and explore univariate data

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#holds all the columns that are measurements from the Lindenmayer study
MEASUREMENT_COLUMNS =  [
    {'col':'total_length', 'string':'Total Length'},
    {'col':'tail_length', 'string' : 'Tail Length'},
    {'col':'head_length', 'string' : 'Head Length'},
    {'col':'skull_width', 'string': 'Skull Width'}, 
    {'col': 'foot_length', 'string': 'Foot Length'},
    {'col': 'eye_width', 'string': 'Eye Width'},
    {'col': 'chest_girth',  'string': 'Chest Girth'},
    {'col':'belly_girth','string':'Belly Girth'}, 
    {'col':'ear_length', 'string':'Ear Length'}
]

def make_hist_distributions(df):
    """
    Creates a 3 x 3 graph with subplots of the distribution of the measurements
    in the data frame
    """
    #make a plot with 9 subplots
    fig, axes = plt.subplots(3, 3, figsize = (15,15), constrained_layout=True)
    #to set the row and columns for the graph
    r = 1
    c = 1
    #now plot each column
    for i, col_dict in enumerate(MEASUREMENT_COLUMNS):
        if i%3 == 0:
            #reached the end of the row, reset column parameter
            c=1
        sns.histplot(x=df[col_dict['col']], ax=axes[r-1,c-1])
        axes[r-1,c-1].set_xlabel(col_dict['string'])
        if c == 3:
            #reached the end of the row, move to the next row
            r +=1
        #move to the next column for the next graph
        c += 1
    #set the title of the graph
    fig.suptitle("Distribution of the Measurements (in mm)")
    plt.show()

def make_boxplot_outliers(df):
    """
    Creates a 3 x 3 graph with subplots of the boxplots of the measurements
    in the data frame
    """
    #make a plot with 9 subplots arranged in a single column
    fig, axes = plt.subplots(3, 3, figsize = (15,18), constrained_layout=True)
    #to set the row and columns for the graph
    r = 1
    c = 1
    for i, col_dict in enumerate(MEASUREMENT_COLUMNS):
        if i%3 == 0:
            #reached the end of the row, reset column parameter
            c=1
        #make the boxplot
        sns.boxplot(data=df, y=col_dict['col'], ax = axes[r-1, c-1])
        #set x and y labels
        axes[r-1, c-1].set_xlabel(col_dict['string'])
        axes[r-1, c-1].set_ylabel('')
        if c == 3:
            #reached the end of the row, move to the next row
            r +=1
        #move to the next column for the next graph
        c += 1
    #set supertitle
    fig.suptitle('Measurement Boxplots Showing Outliers')
    #show plot
    plt.show()

def get_outlier_bounds(s, k):
    """"
    Returns the lower and upper bounds of the passed series object based on the IQR Rule
    Arguments:
        s (pd.Series) : a series object
        k (float) : the IQR rule multiplier
    """
    #calculate iqr
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    #find upper and lower bounds
    upper_bound = q3 + iqr*k
    lower_bound = q1 - iqr*k
    return lower_bound, upper_bound

def get_outliers_df(df):
    """
    Creates a dataframe showing the total number of outliers, the number of lower outliers,
    and the number of upper outliers
    Arguments:
        df (DataFrame) : a dataframe containing the possum data
    """
    #to store outputs
    outputs = []
    for col_dict in MEASUREMENT_COLUMNS:
        # get bounds
        lower, upper = get_outlier_bounds(df[col_dict['col']], 1.5)
        #count the numbers of outliers
        output = {
            'column_name':col_dict['col'],
            'total_outliers': ((df[col_dict['col']] < lower) | (df[col_dict['col']] > upper)).sum(),
            'lower_outliers':(df[col_dict['col']] < lower).sum(),
            'upper_outliers':(df[col_dict['col']] > upper).sum()
        }
        #add the output to the list
        outputs.append(output)
    #return a dataframe
    return pd.DataFrame(outputs)