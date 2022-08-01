# Stephen FitzSimon
# possum regression project explore_target_age.py files
# Contains functions to explore the drivers of the age column

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

#A list of all measurement columns
MEASUREMENT_COLUMNS = [
    'head_length', 
    'skull_width', 
    'total_length',
    'tail_length', 
    'foot_length', 
    'ear_length', 
    'eye_width', 
    'chest_girth', 
    'belly_girth'
]

def make_correlation_plots(df):
    """
    Accepts a possum dataframe, and returns a 3 x 3 of scatter plots with a regression line
    """
    #make a plot with 9 subplots arranged in a single column
    fig, axes = plt.subplots(3, 3, figsize = (15,18), constrained_layout=True)
    #to set the row and columns for the graph
    r = 1
    c = 1
    for i, col in enumerate(MEASUREMENT_COLUMNS):
        if i%3 == 0:
            #reached the end of the row, reset column parameter
            c=1
        #make the regression plot
        sns.regplot(data=df, y = 'age', x=col, line_kws = {'color':'red'}, ax = axes[r-1, c-1])
        #set x and y labels
        axes[r-1, c-1].set_xlabel(col)
        axes[r-1, c-1].set_ylabel('age')
        if c == 3:
            #reached the end of the row, move to the next row
            r +=1
        #move to the next column for the next graph
        c += 1
    #set supertitle
    fig.suptitle('Regression Line For Age and Measurement Columns')
    #show plot
    plt.show()
    
def correlation_test(df, y_col = 'age', alpha = 0.05):
    outputs = []
    for col in MEASUREMENT_COLUMNS:
        corr, p = stats.pearsonr(df[col], df[y_col])
        output = {
            'x_column':col,
            'y_column':y_col,
            'r_value':corr,
            'p_value':p,
            'reject_null': p < alpha
        }
        outputs.append(output)
    return pd.DataFrame(outputs)