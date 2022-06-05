# make_dataset.py

import os, sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join('.', 'src')))
from sklearn.base import BaseEstimator, TransformerMixin
from features.make_features import *


RAW_PATH = '././data/raw'
PROCESSED_PATH = '././data/processed'
INTERIM_PATH = '././data/processed'

def sleep_score_dataset(raw_path: str, interim_path) -> None:
    """Create dataset that contains sleep scores
    
    Parameters
    __________
    raw_path:   str
        Path that contains the raw immutable data
    interim_path:   str
        Path that would hold the processed data

    Returns
    _______
    None
    """

    condition_dir = f'{raw_path}/condition'
    control_dir = f'{raw_path}/control'

    # create condition and control directory in interim folder if it does not exist
    try:
        os.makedirs(f'{interim_path}/condition/')
        os.makedirs(f'{interim_path}/control/')
    except FileExistsError:
        pass

    # create sleep scores for conditions
    for f in os.listdir(condition_dir):
        data = pd.read_csv(f'{condition_dir}/{f}')
        scores = cole_krikpe(data['activity'])
        data['sleep score'] = scores

        # save the dataset in interim
        data.to_csv(f'{interim_path}/condition/{f}', index=None)

    # create sleep scores for controls
    for f in os.listdir(control_dir):
        data = pd.read_csv(f'{control_dir}/{f}')
        scores = cole_krikpe(data['activity'])
        data['sleep score'] = scores

        # save the dataset in interim
        data.to_csv(f'{interim_path}/control/{f}', index=None)

def sleep_interval_dataset(raw_path: str, interim_path: str):
    condition_dir = f'{raw_path}/condition'
    control_dir = f'{raw_path}/control'

    for f in os.listdir(condition_dir):
        data = pd.read_csv(f'{condition_dir}/{f}')
        status = mark_sleep_interval(data['sleep score'])
        data['sleep status'] = status

        # save the dataset in interim
        data.to_csv(f'{interim_path}/condition/{f}', index=None)

    # create sleep scores for controls
    for f in os.listdir(control_dir):
        data = pd.read_csv(f'{control_dir}/{f}')
        status = mark_sleep_interval(data['sleep score'])
        data['sleep status'] = status

        # save the dataset in interim
        data.to_csv(f'{interim_path}/control/{f}', index=None)

def zero_count(series):
    return list(series).count(0)

def nextday(dates):
    for date in dates:
        yield date

def get_summary_data(dir, state=None):
    X = []
    for f in os.listdir(dir):
        data = pd.read_csv(f'{dir}/{f}')
        data['log_activity'] = np.log(data['activity'] + 1)
        dates = data.date.unique()

        for date in nextday(dates):
            data_ext = extractfeatures(data, date)
            X.append(data_ext)

        df = pd.DataFrame(X)
        df['state'] = state

    return df

def extractfeatures(X, date):
    mask = X['date'] == date
    d = {
        'mean_log_activity': X[mask]['log_activity'].mean(),
        'std_log_activity': X[mask]['log_activity'].std(),
        'zero_proportion_activity': zero_count(X[mask]['log_activity'])
    }
    return d

def remove_invalid_data(data):
    """Detects if there is 720 strings of zero and remove them from the dataset"""

    for idx, _ in data.iterrows():
        if (len(data.loc[idx: idx + 719, 'activity']) == 720) and (sum(data.loc[idx: idx + 719, 'activity']) == 0):
            return data.iloc[0: idx, :]

    return data

def act_summary_dataset(raw_path: str, processed_path: str):
    condition_dir = f'{raw_path}/condition'
    control_dir = f'{raw_path}/control'
    condition_df = get_summary_data(condition_dir,1)
    control_df = get_summary_data(control_dir,0)
    condition_df.to_csv(f'{processed_path}/conditions001.csv', index=False)
    control_df.to_csv(f'{processed_path}/controls001.csv', index=False)
        

class SleepParamsAdder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sleep_scores = cole_krikpe(X.activity)
        sleep_intervals = mark_sleep_interval(sleep_scores)
        data = pd.DataFrame(np.c_[X.activity, sleep_intervals, sleep_scores], columns=['activity', 'status', 'score'])

        act_mean = a_mean(data.status, data.activity)
        act_median = a_median(data.status, data.activity)
        act_std = a_std(data.status, data.activity)
        sleep_ef = sleep_efficiency(data.status, data.score)
        total_smin = tsmin(data.status, data.score)
        wake_so = waso(data.status, data.score)
        wake_eps = wake_episode(data.status, data.score)

        data =  np.c_[
            act_mean,
            act_median,
            act_std,
            sleep_ef,
            total_smin,
            wake_so,
            wake_eps,
        ]

        return pd.DataFrame(data, columns=['act_mean', 'act_median', 'act_std', 'sleep_ef', 'total_smin', 'wake_so', 'wake_eps'])

def sleep_params_dataset(raw_path: str, processed_path: str):
    """Given a string of the raw path and processed_path, this function extract the sleep parameters from the activity data of all files
    found in the raw path and saves the result to the processed path for further processing. If the processed_path is None, it returns a tuple
    containing the condition and control data"""
    
    condition_dir = f'{raw_path}/condition'
    control_dir = f'{raw_path}/control'

    control_df = pd.DataFrame(columns=['act_mean', 'act_median',  'act_std', 'sleep_ef', 'total_smin', 'wake_so', 'wake_eps'])
    condition_df = pd.DataFrame(columns=['act_mean', 'act_median',  'act_std', 'sleep_ef', 'total_smin', 'wake_so', 'wake_eps'])

    sleep_params_obj = SleepParamsAdder()

    for f in os.listdir(condition_dir):
        data = pd.read_csv(f'{condition_dir}/{f}')
        data_cleaned = remove_invalid_data(data)
        data_tr = sleep_params_obj.fit_transform(data_cleaned)
        condition_df = condition_df.append(data_tr, ignore_index=False)
        condition_df['state'] = 1

    for f in os.listdir(control_dir):
        data = pd.read_csv(f'{control_dir}/{f}')
        data_cleaned = remove_invalid_data(data)
        data_tr = sleep_params_obj.fit_transform(data_cleaned)
        control_df = control_df.append(data_tr, ignore_index=False)
        control_df['state'] = 0

    if processed_path:
        condition_df.to_csv(f'{processed_path}/conditions002.csv', index=False)
        control_df.to_csv(f'{processed_path}/controls002.csv', index=False)
    else:
        return condition_df, control_df

if __name__ == '__main__':
   # act_summary_dataset(RAW_PATH, PROCESSED_PATH)
   sleep_params_dataset(RAW_PATH, PROCESSED_PATH)

   remove_invalid_data(pd.read_csv(RAW_PATH + '/control/control_27.csv'))
