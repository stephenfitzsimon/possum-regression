# Stephen FitzSimon
# possum regression project explore_target_sex.py files
# Contains functions to prepare the possum dataframe for multivariate exploration
# and modelling of the sex target variable

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
from scipy.stats import ttest_ind, levene

#holds all the columns that are measurements from the Lindenmayer study
MEASUREMENT_COLUMNS = [
    {'col':'total_length', 'string':'Total Length'},
    {'col':'tail_length', 'string' : 'Tail Length'},
    {'col':'head_length', 'string' : 'Head Length'},
    {'col':'skull_width', 'string': 'Skull Width'}, 
    {'col':'foot_length', 'string': 'Foot Length'},
    {'col':'eye_width', 'string': 'Eye Width'},
    {'col':'chest_girth',  'string': 'Chest Girth'},
    {'col':'belly_girth','string':'Belly Girth'}, 
    {'col':'ear_length', 'string':'Ear Length'}
]

def make_boxplot_sex_outliers(df):
    """
    Creates a 3 x 3 graph with subplots of the boxplots of the measurements
    in the data frame hued by the sex column
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
        #calculate all of the satistics and store in a dictionary
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
    #return the dataframe
    return pd.DataFrame(outputs)

def make_pointplot(df):
    """
    Creates a 3 x 3 graph with subplots of the boxplots of the measurements
    in the data frame hued by the sex column
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
    fig, axes = plt.subplots(3, 3, figsize = (15,18), constrained_layout=True)
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
    """
    Performs a two sample t-test for the male and female sets of records
    for every measurement column.  Returns a dataframe with the results
    """
    # holds the outputs
    outputs = []
    #split the dataframe between the males and females
    male_df = df[df['sex']=='male']
    female_df = df[df['sex']=='female']
    for col_dict in MEASUREMENT_COLUMNS:
        #perform a leven test to check if the variances are the same
        stat_levene, p_levene = levene(male_df[col_dict['col']], female_df[col_dict['col']])
        #perform the t-test
        t, p = ttest_ind(male_df[col_dict['col']], female_df[col_dict['col']], equal_var = not (p_levene < alpha))
        #save output to a dictionary
        output = {
            'column_name':col_dict['col'],
            't-stat':t,
            'p-value':p,
            'reject_null':p < alpha
        }
        #add the calculated stats to the list
        outputs.append(output)
    #return as a dataframes
    return pd.DataFrame(outputs)

def make_ratio_dataframe(df):
    """
    Creates a dataframe with the pairs of every column, and a list
    of the column names.
    """
    #make a dataframe to store the ratios
    ratio_df = df.loc[:,'case':'sex']
    #make a series of pairs
    pairs = it.combinations(MEASUREMENT_COLUMNS, 2)
    column_names = []
    for col_one, col_two in pairs:
        column_names.append(f"{col_one['col']}_ratio_{col_two['col']}")
        #make a ratio column for each pair
        ratio_df[f"{col_one['col']}_ratio_{col_two['col']}"] = df[col_one['col']]/df[col_two['col']]
    #return the dataframe
    return ratio_df, column_names

def make_scatterplot_sex_ratios(df):
    """
    Creates a 12 x 3 graph with subplots of the boxplots of the ratio
    of the measurements hued by the sex
    """
    #make a series of pairs
    pairs = it.combinations(MEASUREMENT_COLUMNS, 2)
    #make a plot with 9 subplots arranged in a single column
    fig, axes = plt.subplots(9, 4, figsize = (15,35), constrained_layout=True)
    #to set the row and columns for the graph
    r = 1
    c = 1
    for i, pair in enumerate(pairs):
        if i%4 == 0:
            #reached the end of the row, reset column parameter
            c=1
        #make the boxplot
        sns.scatterplot(data=df, x = pair[0]['col'], y=pair[1]['col'], hue='sex', ax = axes[r-1, c-1])
        #set x and y labels
        axes[r-1, c-1].set_xlabel(pair[0]['string'])
        axes[r-1, c-1].set_ylabel(pair[1]['string'])
        if c == 4:
            #reached the end of the row, move to the next row
            r +=1
        #move to the next column for the next graph
        c += 1
    #set supertitle
    fig.suptitle('Measurement Ratio Boxplots')
    #show plot
    plt.show()

def make_boxplot_sex_ratios(df):
    """
    Creates a 12 x 3 graph with subplots of the scatterplots of the pairs
    of the measurements hued by the sex
    """
    #get the ratio dataframe
    df_ratios, col_names = make_ratio_dataframe(df)
    #make a plot with 9 subplots arranged in a single column
    fig, axes = plt.subplots(9, 4, figsize = (15,35), constrained_layout=True)
    #to set the row and columns for the graph
    r = 1
    c = 1
    for i, col in enumerate(col_names):
        if i%4 == 0:
            #reached the end of the row, reset column parameter
            c=1
        #make the boxplot
        sns.boxplot(data=df_ratios, x = 'sex', y=col, ax = axes[r-1, c-1])
        #set x and y labels
        axes[r-1, c-1].set_xlabel(col)
        axes[r-1, c-1].set_ylabel('')
        if c == 4:
            #reached the end of the row, move to the next row
            r +=1
        #move to the next column for the next graph
        c += 1
    #set supertitle
    fig.suptitle('Measurement Ratio Boxplots')
    #show plot
    plt.show()
    
def hypothesis_two_sample_ttest_ratio(df, alpha = 0.05):
    """
    Performs a two sample t-test for the male and female sets of records
    for every measurement ratio.  Returns a dataframe with the results
    """
    #get the ratio dataframe
    df_ratios, col_names = make_ratio_dataframe(df)
    # holds the outputs
    outputs = []
    #split the dataframe between the males and females
    male_df = df_ratios[df_ratios['sex']=='male']
    female_df = df_ratios[df_ratios['sex']=='female']
    for col in col_names:
        #perform a leven test to check if the variances are the same
        stat_levene, p_levene = levene(male_df[col], female_df[col])
        #perform the t-test
        t, p = ttest_ind(male_df[col], female_df[col], equal_var = not (p_levene < alpha))
        #save output to a dictionary
        output = {
            'ratio_column':col,
            't-stat':t,
            'p-value':p,
            'reject_null':p < alpha
        }
        #add the calculated stats to the list
        outputs.append(output)
    #return as a dataframes
    return pd.DataFrame(outputs)

def make_bodylength_column(df):
    """
    Creates a dataframe with the bodylength column
    """
    #make a dataframe to store the ratios
    bodylength_df = df.copy()
    #make body length column
    bodylength_df['body_length'] = bodylength_df['total_length'] - (bodylength_df['tail_length'] + bodylength_df['head_length'])
    return bodylength_df

def single_two_sample_ttest(df, column_name, alpha = 0.05):
    """
    Performs a single two sample ttest on a column given by column_name
    """
    #to hold test output
    outputs = []
    #split dataframe into male and female
    male_df = df[df['sex']=='male']
    female_df = df[df['sex']=='female']
    #perform a leven test to check if the variances are the same
    stat_levene, p_levene = levene(male_df[column_name], female_df[column_name])
    #perform the t-test
    t, p = ttest_ind(male_df[column_name], female_df[column_name], equal_var = not (p_levene < alpha))
    #save output to a dictionary
    output = {
        'column_name':column_name,
        't-stat':t,
        'p-value':p,
        'reject_null':p < alpha
    }
    #add the calculated stats to the list
    outputs.append(output)
    #return a dataframe
    return pd.DataFrame(outputs)

def single_two_sample_ttest_greater(df, column_name, alpha = 0.05):
    """
    Performs a single two sample ttest on a column given by column_name
    """
    #to hold test output
    outputs = []
    #split dataframe into male and female
    male_df = df[df['sex']=='male']
    female_df = df[df['sex']=='female']
    #perform a leven test to check if the variances are the same
    stat_levene, p_levene = levene(male_df[column_name], female_df[column_name])
    #perform the t-test
    t, p = ttest_ind(male_df[column_name], female_df[column_name], equal_var = not (p_levene < alpha))
    #save output to a dictionary
    output = {
        'column_name':column_name,
        't-stat':t,
        'p-value':p,
        'reject_null':p < alpha
    }
    #add the calculated stats to the list
    outputs.append(output)
    #return a dataframe
    return pd.DataFrame(outputs)