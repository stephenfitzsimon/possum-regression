# Stephen FitzSimon
# possum regression project model_target_age.py files
# Contains functions to prepare a dataframe for modeling
# and model the drivers of the age column

from math import sqrt

import pandas as pd
from scipy import stats
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

### GLOBAL CONSTANTS

# ensures that the models are always the same as in the notebook
RAND_SEED = 1729 #taxicab number

# holds all of the names of measurement columns
MEASUREMENT_COLUMNS = [
    'total_length',
    'tail_length',
    'head_length',
    'skull_width',
    'foot_length',
    'eye_width',
    'chest_girth',
    'belly_girth',
    'ear_length'
]

# holds columns correlated with age
CORRELATED_COLUMNS = [
    'chest_girth',
    'belly_girth',
    'head_length',
    'total_length'
]

#holds the model strategy types
MODEL_STRATEGIES = ['all_columns', 'correlated_columns_only']

#sets up the model strategies
MODEL_STRATEGY_DICTIONARY = {
    'all_columns':MEASUREMENT_COLUMNS,
    'correlated_columns_only':CORRELATED_COLUMNS
}

#columns that will not be used in models
DROP_COLUMNS = [
    'trap_site', 
    'state', 
    'latitude', 
    'longitude', 
    'elevation',
    'sex'
]

MODELS = [
    {
        'name':'RadiusNeighborsRegressor',
        'model':RadiusNeighborsRegressor()
    },
    {
        'name':'KNeighborsRegressor',
        'model':KNeighborsRegressor()
    },
    {
        'name':'TweedieRegressor',
        'model':TweedieRegressor()
    },
    {
        'name':'LinearRegression',
        'model':LinearRegression()
    },
    {
        'name':'RandomForestRegressor',
        'model':RandomForestRegressor()
    }
]

### MODEL MAKERS

def test_model(train, validate, test):
    X_train, y_train, X_val, y_val = prepare_train_validate(train, 
                                                            validate, 
                                                            MODEL_STRATEGY_DICTIONARY['all_columns']
                                                           )
    X_test, y_test = prepare_test(train, 
                                  test, 
                                  MODEL_STRATEGY_DICTIONARY['all_columns']
                                 )
    output, y_test, model = make_radius_neighbor_model_test(X_train, y_train, X_val, y_val, X_test, y_test)
    y_test['residual'] = y_test['predicted'] - y_test['age']
    return pd.DataFrame(output), y_test, model

def make_radius_neighbor_model_test(X_train, y_train, X_val, y_val, X_test, y_test):
    model = RadiusNeighborsRegressor()
    model = model.fit(X_train, y_train['age'])
    y_train['predicted'] = model.predict(X_train)
    y_val['predicted'] = model.predict(X_val)
    y_test['predicted'] = model.predict(X_test)
    output = evaluate_test_model([{'set':'train', 'df':y_train},
                                  {'set':'validate', 'df':y_val},
                                  {'set':'test', 'df':y_test}
                                 ])
    return output, y_test, model

def model_maker(train, validate):
    strategy_column_name = 'modeling_strategy'
    outputs = [] + make_baseline_model(train, validate, return_df = False)
    train_predictions = train[['case', 'age']].copy()
    validate_predictions = validate[['case', 'age']].copy()
    for strategy in MODEL_STRATEGIES:
        X_train, y_train, X_val, y_val = prepare_train_validate(train, validate, MODEL_STRATEGY_DICTIONARY[strategy])
        for model_item in MODELS:
            output, train_predict, validate_predict = make_model(X_train, y_train, X_val, y_val, model_item)
            output['model_strategy'] = strategy
            outputs.append(output)
            model_name = model_item['name']
            train_predictions[f'{model_name}_{strategy}'] = train_predict['predicted']
            validate_predictions[f'{model_name}_{strategy}'] = validate_predict['predicted']
    return pd.DataFrame(outputs), train_predictions, validate_predictions    
               
def make_model(X_train, y_train, X_val, y_val, model_item):
    model_item['model'] = model_item['model'].fit(X_train, y_train['age'])
    y_train['predicted'] = model_item['model'].predict(X_train)
    y_val['predicted'] = model_item['model'].predict(X_val)
    output = evaluate_train_validate_model(y_train, y_val, model_item['name'])
    return output, y_train, y_val

def make_baseline_model(train, validate, return_df = True):
    """
    Creates two baseline model, one based on mean and one based on median.  Returns a dataframe
    Containing their evaluation metrics.
    """
    outputs = []
    baseline_model_mean = train.loc[:, ['case', 'age']]
    baseline_model_median = train.loc[:, ['case', 'age']]
    baseline_model_val_mean = validate.loc[:, ['case', 'age']]
    baseline_model_val_median = validate.loc[:, ['case', 'age']]
    baseline_model_mean['predicted'] = train.age.mean()
    baseline_model_median['predicted'] = train.age.median()
    baseline_model_val_mean['predicted'] = train.age.mean()
    baseline_model_val_median['predicted'] = train.age.median()
    output_mean = evaluate_train_validate_model(baseline_model_mean, baseline_model_val_mean, 'baseline_mean')
    output_median = evaluate_train_validate_model(baseline_model_median, baseline_model_val_median, 'baseline_median')
    outputs.append(output_mean)
    outputs.append(output_median)
    if return_df:
        return pd.DataFrame(outputs)
    else:
        return outputs

def evaluate_test_model(y_values):
    outputs = []
    for y_predict in y_values:
        predictions_df = y_predict['df']
        mse2 = metrics.mean_squared_error(predictions_df['age'], predictions_df['predicted'])
        evs = metrics.explained_variance_score(predictions_df['age'], predictions_df['predicted'])
        output = {
            'dataset':y_predict['set'],
            'RMSE':sqrt(mse2),
            'explained_variance':evs
        }
        outputs.append(output)
    return outputs
        
def evaluate_train_validate_model(model_train, model_validate, model_name):
    #calculate the metrics to evaluate
    mse2_train = metrics.mean_squared_error(model_train['age'], model_train['predicted'])
    mse2_validate = metrics.mean_squared_error(model_validate['age'], model_validate['predicted'])
    r2_train = metrics.r2_score(model_train['age'], model_train['predicted'])
    r2_validate = metrics.r2_score(model_validate['age'], model_validate['predicted'])
    evs_train = metrics.explained_variance_score(model_train['age'], model_train['predicted'])
    evs_validate = metrics.explained_variance_score(model_validate['age'], model_validate['predicted'])
    #store in a dictionary and return to call
    model_metrics = {
        'model':model_name,
        'train_RMSE':sqrt(mse2_train),
        'train_r2':r2_train,
        'train_explained_variance':evs_train,
        'validate_RMSE':sqrt(mse2_validate),
        'validate_r2':r2_validate,
        'validate_explained_variance':evs_validate
    }
    return model_metrics
    
### FLOW CONTROL FUNCTIONS

def prepare_train_validate(train, validate, columns_to_scale):
    """
    Accepts train and validate dataframes and returns
    their X and y versions ready to pass into a model
    """
    #drop unnecessary columns
    train = drop_columns(train)
    validate = drop_columns(validate)
    return make_X_and_y(train, validate, columns_to_scale)

def prepare_test(train, test, columns_to_scale):
    #drop unnecessary columns
    test = drop_columns(test)
    _, _, X_test, y_test = make_X_and_y(train, test, columns_to_scale)
    return X_test, y_test

def drop_columns(df):
    """
    Drops the columns that will not be modeled on
    """
    #drop columns and return the dataframe
    return df.drop(columns = DROP_COLUMNS)

def make_X_and_y(train, validate, columns_to_scale):
    """
    Scales the columns for modeling
    """
    #make the scaler
    scaler = StandardScaler()
    #fit the scaler and transform the columns for X_train
    X_train = scaler.fit_transform(train[columns_to_scale])
    #transform the columns for X_validate
    X_val = scaler.transform(validate[columns_to_scale])
    #make y dataframes
    y_train = train[['case', 'age']]
    y_val = validate[['case', 'age']]
    return X_train, y_train, X_val, y_val