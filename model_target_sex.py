# Stephen FitzSimon
# possum regression project model_target_sex.py files
# Contains functions to prepare the possum dataframe for modelling of 
# the sex target variable

import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

#model modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn.neighbors as neigh

### GLOBAL CONSTANTS

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

# columns that are used in modeling but are
# not ratio or measurement columns
OTHER_MODEL_COLUMNS = [
    'age', 
    'sex'
]

#holds the model strategy types
MODEL_STRATEGIES = ['measurement_only', 'ratio_only', 'measurement_ratio']

#sets up the model strategies
MODEL_STRATEGY_DICTIONARY = {
    'measurement_only':MEASUREMENT_COLUMNS,
    'measurement_ratio': MEASUREMENT_COLUMNS + RATIO_COLUMNS,
    'ratio_only': RATIO_COLUMNS
}

#columns that will not be used in models
DROP_COLUMNS = [
    'trap_site', 
    'state', 
    'latitude', 
    'longitude', 
    'elevation',
    'age'
]

### MODEL MAKERS

def test_model(train, validate, test, baseline_acc):
    X_train, y_train, X_val, y_val = prepare_train_validate(train, validate, MODEL_STRATEGY_DICTIONARY['measurement_ratio'])
    X_test, y_test = prepare_test(train, test, MODEL_STRATEGY_DICTIONARY['measurement_ratio'])
    output, y_test = make_radius_neighbor_model_test(X_train, y_train, X_val, y_val, X_test, y_test, baseline_acc)
    return pd.DataFrame([output]), y_test

def model_maker(train, validate, baseline_acc):
    strategy_column_name = 'modeling_strategy'
    outputs = []
    for strategy in MODEL_STRATEGIES:
        X_train, y_train, X_val, y_val = prepare_train_validate(train, validate, MODEL_STRATEGY_DICTIONARY[strategy])
        output_log_reg = make_log_reg_model(X_train, y_train, X_val, y_val, baseline_acc)
        output_log_reg[strategy_column_name] = strategy
        outputs.append(output_log_reg)
        output_svm = make_svm_model(X_train, y_train, X_val, y_val, baseline_acc)
        output_svm[strategy_column_name] = strategy
        outputs.append(output_svm)
        output_knn = make_knn_model(X_train, y_train, X_val, y_val, baseline_acc)
        output_knn[strategy_column_name] = strategy
        outputs.append(output_knn)
        output_rn = make_radius_neighbor_model(X_train, y_train, X_val, y_val, baseline_acc)
        output_rn[strategy_column_name] = strategy
        outputs.append(output_rn)
        output_nc = make_nearest_centroid_model(X_train, y_train, X_val, y_val, baseline_acc)
        output_nc[strategy_column_name] = strategy
        outputs.append(output_nc)
    return pd.DataFrame(outputs)

def make_baseline_model(train, validate):
    baseline_model = train.loc[:, ['case', 'sex']]
    baseline_model_val = validate.loc[:, ['case', 'sex']]
    baseline_model['predicted'] = train.sex.mode().to_list()[0]
    baseline_model_val['predicted'] = train.sex.mode().to_list()[0]
    metrics_dict = metrics.classification_report(train['sex'], baseline_model['predicted'], output_dict=True, zero_division = True)
    metrics_dict_val = metrics.classification_report(validate['sex'], baseline_model_val['predicted'], output_dict=True, zero_division = True)
    output = {
        'model':'Baseline Model',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
    }
    return pd.DataFrame([output]), metrics_dict['accuracy']

def make_log_reg_model(X_train, y_train, X_val, y_val, baseline_acc, return_report = False):
    lm = LogisticRegression(max_iter = 100, random_state = RAND_SEED).fit(X_train, y_train['sex'])
    y_train['predicted'] = lm.predict(X_train)
    y_val['predicted'] = lm.predict(X_val)
    metrics_dict = metrics.classification_report(y_train['sex'], y_train['predicted'], output_dict=True)
    metrics_dict_val = metrics.classification_report(y_val['sex'], y_val['predicted'], output_dict=True)
    output = {
        'model':'Logistic Regression',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'accuracy_change': metrics_dict['accuracy'] - metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    if return_report:
        return metrics_dict, metrics_dict_val
    else:
        return output

def make_svm_model(X_train, y_train, X_val, y_val, baseline_acc, return_report = False):
    svm = SVC(random_state = RAND_SEED).fit(X_train, y_train['sex'])
    y_train['predicted'] = svm.predict(X_train)
    y_val['predicted'] = svm.predict(X_val)
    metrics_dict = metrics.classification_report(y_train['sex'], y_train['predicted'], output_dict=True)
    metrics_dict_val = metrics.classification_report(y_val['sex'], y_val['predicted'], output_dict=True)
    output = {
        'model':'Support Vector Classification',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'accuracy_change': metrics_dict['accuracy'] - metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    if return_report:
        return metrics_dict, metrics_dict_val
    else:
        return output

def make_knn_model(X_train, y_train, X_val, y_val, baseline_acc, return_report = False):
    knn = neigh.KNeighborsClassifier().fit(X_train, y_train['sex'])
    y_train['predicted'] = knn.predict(X_train)
    y_val['predicted'] = knn.predict(X_val)
    metrics_dict = metrics.classification_report(y_train['sex'], y_train['predicted'], output_dict=True)
    metrics_dict_val = metrics.classification_report(y_val['sex'], y_val['predicted'], output_dict=True)
    output = {
        'model':'K-Nearest Neighbors Classification',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'accuracy_change': metrics_dict['accuracy'] - metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    if return_report:
        return metrics_dict, metrics_dict_val
    else:
        return output

def make_radius_neighbor_model(X_train, y_train, X_val, y_val, baseline_acc, return_report = False):
    rn = neigh.RadiusNeighborsClassifier(radius = 2.5).fit(X_train, y_train['sex'])
    y_train['predicted'] = rn.predict(X_train)
    y_val['predicted'] = rn.predict(X_val)
    metrics_dict = metrics.classification_report(y_train['sex'], y_train['predicted'], output_dict=True)
    metrics_dict_val = metrics.classification_report(y_val['sex'], y_val['predicted'], output_dict=True)
    output = {
        'model':'Radius Neighbors Classification',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'accuracy_change': metrics_dict['accuracy'] - metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    if return_report:
        return metrics_dict, metrics_dict_val
    else:
        return output
    
def make_radius_neighbor_model_test(X_train, y_train, X_val, y_val, X_test, y_test, baseline_acc):
    rn = neigh.RadiusNeighborsClassifier(radius = 2.5).fit(X_train, y_train['sex'])
    y_train['predicted'] = rn.predict(X_train)
    y_val['predicted'] = rn.predict(X_val)
    y_test['predicted'] = rn.predict(X_test)
    y_probs = rn.predict_proba(X_test)
    y_test['probability_male'] = y_probs[:,1]
    y_test['probability_female'] = y_probs[:,0]
    metrics_dict = metrics.classification_report(y_train['sex'], y_train['predicted'], output_dict=True)
    metrics_dict_val = metrics.classification_report(y_val['sex'], y_val['predicted'], output_dict=True)
    metrics_dict_test = metrics.classification_report(y_test['sex'], y_test['predicted'], output_dict=True)
    output = {
        'model':'Radius Neighbors Classification',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'test_accuracy': metrics_dict_test['accuracy'],
        'better_than_baseline':metrics_dict_test['accuracy'] > baseline_acc
    }
    return output, y_test

def make_nearest_centroid_model(X_train, y_train, X_val, y_val, baseline_acc, return_report = False):
    nc = neigh.NearestCentroid().fit(X_train, y_train['sex'])
    y_train['predicted'] = nc.predict(X_train)
    y_val['predicted'] = nc.predict(X_val)
    metrics_dict = metrics.classification_report(y_train['sex'], y_train['predicted'], output_dict=True)
    metrics_dict_val = metrics.classification_report(y_val['sex'], y_val['predicted'], output_dict=True)
    output = {
        'model':'Nearest Centroid Classification',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'accuracy_change': metrics_dict['accuracy'] - metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    if return_report:
        return metrics_dict, metrics_dict_val
    else:
        return output

### FLOW CONTROL FUNCTIONS

def prepare_train_validate(train, validate, columns_to_scale):
    """
    Accepts train and validate dataframes and returns
    their X and y versions ready to pass into a model
    """
    #make columns to be modeled on
    train = make_modeling_columns(train)
    validate = make_modeling_columns(validate)
    #drop unnecessary columns
    train = drop_columns(train)
    validate = drop_columns(validate)
    return make_X_and_y(train, validate, columns_to_scale)

def prepare_test(train, test, columns_to_scale):
    #make columns to be modeled on
    test = make_modeling_columns(test)
    #drop unnecessary columns
    test = drop_columns(test)    
    _, _, X_test, y_test = make_X_and_y(train, test, columns_to_scale)
    return X_test, y_test

### PREPARATION FUNCTIONS

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
    y_train = train[['case', 'sex']]
    y_val = validate[['case', 'sex']]
    return X_train, y_train, X_val, y_val