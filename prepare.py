# Stephen FitzSimon
# possum regression project prepare.py files
# Contains functions to prepare the possum dataframe for multivariate exploration
# and modelling

import pandas as pd
from sklearn.model_selection import train_test_split

RAND_SEED = 1729

def split_data(df):
    '''Splits the possum dataframe into train, test and validate subsets
    Args:
        df (DataFrame) : dataframe to split
    Return:
        train, test, validate (DataFrame) :  dataframes split from the original dataframe
    '''
    #make train and test
    train, validate = train_test_split(df, train_size = 0.7, stratify=df[['sex']], random_state=RAND_SEED)
    #make validate
    validate, test = train_test_split(train, train_size = 0.5, stratify=train[['sex']], random_state=RAND_SEED)
    return train, validate, test

def make_sex_distribution_df(train, validate, test):
    """
    Makes a dataframe showing the distribution of the sex column across the split
    data
    """
    #make a list of dictionaries
    calculations = [
        {
            'dataset':'train', #dataset name
            'proportion_male':train.sex.value_counts(normalize=True)['male'], #get proportion of males
            'proportion_female':train.sex.value_counts(normalize=True)['female'], #get proportion of females
            'total_male':train.sex.value_counts()['male'], #get the count of males
            'total_female':train.sex.value_counts()['female'] #get the count of females
        },
        {
            'dataset':'validate',
            'proportion_male':validate.sex.value_counts(normalize=True)['male'],
            'proportion_female':validate.sex.value_counts(normalize=True)['female'],
            'total_male':validate.sex.value_counts()['male'],
            'total_female':validate.sex.value_counts()['female']
        },
        {
            'dataset':'test',
            'proportion_male':test.sex.value_counts(normalize=True)['male'],
            'proportion_female':test.sex.value_counts(normalize=True)['female'],
            'total_male':test.sex.value_counts()['male'],
            'total_female':test.sex.value_counts()['female']
        }
    ]
    #return as a dataframe
    return pd.DataFrame(calculations)