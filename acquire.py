import pandas as pd

FILENAME = 'possum.csv'

def make_dataset():
    df = get_dataset()
    df = rename_columns(df)
    df = map_trap_site_names(df)
    df = map_state_names(df)
    df = convert_centimeters(df)
    return df

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
    trap_site_map = {
        1:'Cambarville',
        2:'Bellbird',
        3:'Whian Whian State Forest',
        4:'Byrangery Reserve',
        5:'Conondale Ranges',
        6:'Bulburin State Forest',
        7:'Allyn River Forest Park'
    }
    df['trap_site'] = df['trap_site'].map(trap_site_map)
    return df

def map_state_names(df):
    """
    Maps the state name to a string (information from the original paper
    and from the DAAG module)
    """
    state_map = {
        'Cambarville':'Victoria',
        'Bellbird':'Victoria',
        'Whian Whian State Forest':'New South Wales',
        'Byrangery Reserve':'New South Wales',
        'Conondale Ranges':'Queensland',
        'Bulburin State Forest':'Queensland',
        'Allyn River Forest Park':'New South Wales'
    }
    df['state'] = df['trap_site'].map(state_map)
    return df

def convert_centimeters(df):
    cm_columns = ['total_length', 'tail_length', 'chest_girth', 'belly_girth']
    for col in cm_columns:
        df[col] = df[col]*10
    return df