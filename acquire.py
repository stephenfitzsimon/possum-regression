# Stephen FitzSimon
# possum regression project acquire.py files
# Contains functions to acquire and prepare the possum data frame

import pandas as pd
import geopandas as gpd

FILENAME = 'possum.csv'

def make_dataset():
    """
    This is a flow control function to prepare the dataset
    """
    df = get_dataset()
    df = rename_columns(df)
    df = map_trap_site_names(df)
    df = map_state_names(df)
    df = convert_centimeters(df)
    df = make_geo_columns(df)
    df = make_sex_column(df)
    return df.dropna()

def get_dataset():
    """
    Gets the dataset from the filename constant.  
    Returns: 
        df (dataframe) : dataframe containing possum data
    """
    # try to access the file
    try:
        df = pd.read_csv(FILENAME)
        # file accessed, return as a dataframe
        return df
    except Exception as e:
        # error warning, print to user
        print(e)
        print(f'Please download the dataset and name it {FILENAME} from https://www.kaggle.com/datasets/abrambeyer/openintro-possum')
        
def rename_columns(df):
    """
    Renames the columns so that they are more human readable
    """
    #dictionary to map the column old names to new names
    col_rename = {
        'site':'trap_site',
        'Pop':'state',
        'hdlngth':'head_length',
        'skullw':'skull_width',
        'totlngth':'total_length',
        'taill': 'tail_length',
        'footlgth':'foot_length',
        'earconch':'ear_length',
        'eye':'eye_width',
        'chest':'chest_girth',
        'belly':'belly_girth'
    }
    #rename and return
    return df.rename(columns = col_rename)

def map_trap_site_names(df):
    """
    Maps the numeric trap_site column to the proper strings of 
    trap locations (information from the original paper and from the
    original DAAG module)
    """
    #dictionary to map the values
    trap_site_map = {
        1:'Cambarville',
        2:'Bellbird',
        3:'Whian Whian State Forest',
        4:'Byrangery Reserve',
        5:'Conondale Ranges',
        6:'Bulburin State Forest',
        7:'Allyn River Forest Park'
    }
    #make the column
    df['trap_site'] = df['trap_site'].map(trap_site_map)
    return df

def map_state_names(df):
    """
    Maps the state name to a string (information from the original paper
    and from the DAAG module)
    """
    # dictionary to map the values
    state_map = {
        'Cambarville':'Victoria',
        'Bellbird':'Victoria',
        'Whian Whian State Forest':'New South Wales',
        'Byrangery Reserve':'New South Wales',
        'Conondale Ranges':'Queensland',
        'Bulburin State Forest':'Queensland',
        'Allyn River Forest Park':'New South Wales'
    }
    #make the column
    df['state'] = df['trap_site'].map(state_map)
    return df

def convert_centimeters(df):
    """
    Converts the centimeter columns to milimeters
    """
    #columns that are centimeters
    cm_columns = ['total_length', 'tail_length', 'chest_girth', 'belly_girth']
    for col in cm_columns:
        #make each column milimeters by multiplying by 10
        df[col] = df[col]*10
    return df

def make_geo_columns(df):
    """
    Makes the latitude, longitude, and elevation.  Data from Lindenmayer and 
    DAAG dataset on CRAN
    """
    #dictionaries to map the values
    lat_map = {
        'Cambarville':-37.55,
        'Bellbird':-37.616667,
        'Allyn River Forest Park':-32.116667,
        'Whian Whian State Forest':-28.616667,
        'Byrangery Reserve':-28.616667,
        'Conondale Ranges':-26.433333,
        'Bulburin State Forest':-24.55
    }
    long_map = {
        'Cambarville':145.8833,
        'Bellbird':148.8,
        'Allyn River Forest Park':151.466667,
        'Whian Whian State Forest':153.333333,
        'Byrangery Reserve':153.416667,
        'Conondale Ranges':152.583333,
        'Bulburin State Forest':151.466667
    }
    elev_map = {
        'Cambarville':800,
        'Bellbird':300,
        'Allyn River Forest Park':300,
        'Whian Whian State Forest':400,
        'Byrangery Reserve':200,
        'Conondale Ranges':400,
        'Bulburin State Forest':600
    }
    #make the columns
    df['latitude'] = df['trap_site'].map(lat_map)
    df['longitude'] = df['trap_site'].map(long_map)
    df['elevation'] = df['trap_site'].map(elev_map)
    return df

def make_sex_column(df):
    sex_map = {
        'm':'male',
        'f':'female'
    }
    df['sex'] = df['sex'].map(sex_map)
    return df