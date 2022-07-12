# Stephen FitzSimon
# possum regression project explore_target_sex.py files
# Contains functions to prepare the possum dataframe for multivariate exploration
# and modelling of the sex target variable

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def make_boxplot_sex_outliers(df):
    """
    Creates a 3 x 3 graph with subplots of the boxplots of the measurements
    in the data frame hued by the sex column
    """
    #make a plot with 9 subplots arranged in a single column
    fig, axes = plt.subplots(3, 3, figsize = (15,18))
    #columns to plot
    columns_to_plot = [
        {'col':'total_length', 'string':'Total Length'},
        {'col':'tail_length', 'string' : 'Tail Length'},
        {'col':'head_length', 'string' : 'Head Length'},
        {'col':'skull_width', 'string': 'Skull Width'}, 
        {'col': 'foot_length', 'string': 'Foot Length'},
        {'col': 'eye_width', 'string': 'Eye Width'},
        {'col': 'chest_girth',  'string': 'Chest Girth'},
        {'col':'belly_girth','string':'Belly Girth'}, 
        { 'col':'ear_length', 'string':'Ear Length'}
    ]
    #to set the row and columns for the graph
    r = 1
    c = 1
    for i, col_dict in enumerate(columns_to_plot):
        if i%3 == 0:
            #reached the end of the row, reset column parameter
            c=1
        #make the boxplot
        sns.boxplot(data=df, x = 'sex', y=col_dict['col'], ax = axes[r-1, c-1])
        #set x and y labels
        axes[r-1, c-1].set_xlabel(col_dict['string'])
        axes[r-1, c-1].set_ylabel('')
        if c == 3:
            #reached the end of the row, move to the next row
            r +=1
        #move to the next column for the next graph
        c += 1
    #set supertitle
    fig.suptitle('Measurement Boxplots Showing Sexual Dimorphism')
    #show plot
    plt.show()