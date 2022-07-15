# Stephen FitzSimon
# possum regression project explore_target_sex.py files
# Contains functions to prepare the possum dataframe for multivariate exploration
# and modelling of the sex target variable

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind, levene

#holds all the columns that are measurements from the Lindenmayer study
MEASUREMENT_COLUMNS = [
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

def make_boxplot_sex_outliers(df):
    """
    Creates a 3 x 3 graph with subplots of the boxplots of the measurements
    in the data frame hued by the sex column
    """
    #make a plot with 9 subplots arranged in a single column
    fig, axes = plt.subplots(3, 3, figsize = (15,18))
    #to set the row and columns for the graph
    r = 1
    c = 1
    for i, col_dict in enumerate(MEASUREMENT_COLUMNS):
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
    
def compare_stats(df):
    """
    Makes a dataframe that shows the mean for male and
    female records
    Arguments:
        df (DataFrame) : a possum dataframe
    """
    #to store the calculations
    outputs = []
    for col_dict in MEASUREMENT_COLUMNS:
        output = {
            'column_name':col_dict['col'],
            'male_mean':df[df['sex']=='male'][col_dict['col']].mean(),
            'female_mean':df[df['sex']=='female'][col_dict['col']].mean(),
            'difference_mean': abs(df[df['sex']=='male'][col_dict['col']].mean() - df[df['sex']=='female'][col_dict['col']].mean()),
            'male_median':df[df['sex']=='male'][col_dict['col']].median(),
            'female_median':df[df['sex']=='female'][col_dict['col']].median(),
            'difference_median': abs(df[df['sex']=='male'][col_dict['col']].median() - df[df['sex']=='female'][col_dict['col']].median()),
            'male_std':df[df['sex']=='male'][col_dict['col']].std(),
            'female_std':df[df['sex']=='female'][col_dict['col']].std()
        }
        outputs.append(output)
    return pd.DataFrame(outputs)

def make_pointplot(df):
    """
    Creates a 3 x 3 graph with subplots of the boxplots of the measurements
    in the data frame hued by the sex column
    """
    #make a plot with 9 subplots arranged in a single column
    fig, axes = plt.subplots(3, 3, figsize = (15,18))
    #to set the row and columns for the graph
    r = 1
    c = 1
    for i, col_dict in enumerate(MEASUREMENT_COLUMNS):
        if i%3 == 0:
            #reached the end of the row, reset column parameter
            c=1
        #make the boxplot
        sns.pointplot(data=df, x = 'sex', y=col_dict['col'], ax = axes[r-1, c-1])
        #set x and y labels
        axes[r-1, c-1].set_xlabel(col_dict['string'])
        axes[r-1, c-1].set_ylabel('')
        if c == 3:
            #reached the end of the row, move to the next row
            r +=1
        #move to the next column for the next graph
        c += 1
    #set supertitle
    fig.suptitle('Measurement Pointplot Showing Sexual Dimorphism')
    #show plot
    plt.show()
    
def make_histograms_by_sex(df):
    """
    Creates a 3 x 3 graph with subplots of the boxplots of the measurements
    in the data frame hued by the sex column
    """
    #make a plot with 9 subplots arranged in a single column
    fig, axes = plt.subplots(3, 3, figsize = (15,18))
    #to set the row and columns for the graph
    r = 1
    c = 1
    for i, col_dict in enumerate(MEASUREMENT_COLUMNS):
        if i%3 == 0:
            #reached the end of the row, reset column parameter
            c=1
        #make the boxplot
        sns.histplot(data=df, x =col_dict['col'], hue='sex', kde=True, ax = axes[r-1, c-1])
        #set x and y labels
        axes[r-1, c-1].set_xlabel(col_dict['string'])
        axes[r-1, c-1].set_ylabel('')
        if c == 3:
            #reached the end of the row, move to the next row
            r +=1
        #move to the next column for the next graph
        c += 1
    #set supertitle
    fig.suptitle('Measurement Distributions Showing Sexual Dimorphism')
    #show plot
    plt.show()
    
def hypothesis_two_sample_ttest(df, alpha = 0.05):
    outputs = []
    male_df = df[df['sex']=='male']
    female_df = df[df['sex']=='female']
    for col_dict in MEASUREMENT_COLUMNS:
        stat_levene, p_levene = levene(male_df[col_dict['col']], female_df[col_dict['col']])
        t, p = ttest_ind(male_df[col_dict['col']], female_df[col_dict['col']], equal_var = not (p_levene < alpha))
        output = {
            'column_name':col_dict['col'],
            't-stat':t,
            'p-value':p,
            'reject_null':p < alpha
        }
        outputs.append(output)
    return pd.DataFrame(outputs)

