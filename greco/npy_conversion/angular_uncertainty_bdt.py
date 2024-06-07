r'''Run this script on GRECO data files after the model has been
trained. This will add the extra columns as well as load the 
model and perform the regression'''
import os
import numpy as np
from glob import glob
import pickle
import argparse, sys
    
import pandas as pd
from numpy.lib.recfunctions import append_fields

df_feature_cols = ['nstring', 'nchannel', 'zen', 'logE', 'cascade energy',
         'monopod zen', 'pidDeltaLLH', 'pidPeglegLLH', 'pidMonopodLLH',
         'pidLength', 'monopod pegleg dpsi']

def load_model():
    r'''
    Load already trained cross validated model. Default kwargs are
    those found to result in best model previously
    Originally from here:
        https://github.com/apizzuto/Novae/blob/master/random_forest/load_model.py
    Model is identical to the one stored at
        /data/user/apizzuto/Nova/RandomForests/v2.5/GridSearchResults_logSeparation_True_bootstrap_True_minsamples_100
    Returns:
    --------
    model: sklearn.ensemble.RandomForestRegressor
        Best model from a cross validated hyperparameter grid search
    '''
    path = '/data/ana/PointSource/GRECO_online/regressor_logSeparation_True_bootstrap_True_minsamples_100.pckl'
    cv = pickle.load(open(path, 'rb'))
    best_model = cv.best_estimator_
    return best_model


def clean_data(data):
    r'''
    Given data as a .npy array with all relevant columns except
    circularized sigma, prepare data to be passed to model to 
    calculate circularized sigma
    Parameters:
    -----------
    data: np.ndarray
        formatted numpy array (skylab format-like)
    Returns:
    --------
    X: matrix
        Array to pass directly to model to make predictions
    '''
    
    if 'index' in data.dtype.names:
        df = pd.DataFrame.from_records(data, index='index')
    else:
        df = pd.DataFrame.from_records(data)
    df = df.replace([np.inf, -np.inf], np.nan)#.dropna(axis=0)
    data = df.to_records()
    
    prediction_events = data.copy()
    usable_dtype = np.dtype([('index', '<i8'), ('run', '<i4'), ('event', '<i4'), ('subevent', '<i4'), 
                             ('nstring', '<i4'), ('nchannel', '<i4'), ('time', '<f8'), 
                             ('ra', '<f4'), ('dec', '<f4'), ('azi', '<f4'), ('zen', '<f4'), ('angErr', '<f4'), 
                             ('logE', '<f4'), ('cascade_energy', '<f4'), ('monopod_azi', '<f4'), ('monopod_zen', '<f4'), 
                             ('pidDeltaLLH', '<f4'), ('pidPeglegLLH', '<f4'), ('pidMonopodLLH', '<f4'), ('pidLength', '<f4'), 
                             ('monopod_ra', '<f8'), ('monopod_dec', '<f8'), ('monopod_pegleg_dpsi', '<f8')])
    if prediction_events.dtype != usable_dtype:
        values = [prediction_events[key] for key in usable_dtype.names]
        values = np.array(values).T
        values = [tuple(x) for x in values]
        prediction_events = np.array(values, dtype=usable_dtype)
    
    events_df = pd.DataFrame.from_dict(prediction_events)
    old_names = events_df.columns
    new_names = [on.replace('_', ' ') for on in old_names]
    events_df.columns = new_names
    events_df = events_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    scaled_events = events_df.copy()
    scaled_events['monopod zen'] = np.cos(events_df['monopod zen'])
    scaled_events['zen'] = np.cos(events_df['zen'])
    scaled_events['pidMonopodLLH'] = np.log10(events_df['pidMonopodLLH'])
    events_df = scaled_events.copy()
    X = events_df[df_feature_cols].values
    return X

def predict_uncertainty(data):
    r'''
    Load and make prediction with a trained model
    Parameters:
    -----------
    X: np.ndarray
        Data cleaned using functions above
    Returns:
    --------
    predictions for opening angle (IN RADIANS)
    '''
    model = load_model() #no arguments defaults to optimal model
    X = clean_data(data)
    y_pred = model.predict(X)
    y_pred = np.power(10., y_pred)
    return y_pred
