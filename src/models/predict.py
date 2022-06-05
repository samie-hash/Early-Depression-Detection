# predict.py
import os, sys
sys.path.append(os.path.abspath(os.path.join('.', 'src')))

from data.make_dataset import SleepParamsAdder, remove_invalid_data
import numpy as np
import pandas as pd
import pickle

model_path = os.getcwd() + '/src/models/model_2022-05-26 14_07_59.825203.pkl'
model = pickle.load(open(model_path, 'rb'))
print('Model sucessfully loaded')

def preprocess(data):
    sleep_params_obj = SleepParamsAdder()
    data_cleaned = remove_invalid_data(data)
    data_tr = sleep_params_obj.fit_transform(data_cleaned)

    return data_tr

def make_prediction(data):
    """Make prediction on a patient activity data"""

    prepared_data = preprocess(data)
    predictions_proba = model.predict_proba(prepared_data)
    predictions = [1 if prob[1] >= 0.40 else 0 for prob in predictions_proba]
    cum_prediction = np.sum(predictions) / len(predictions)

    if cum_prediction > 0.55:
        return 1, cum_prediction
    else:
        return 0, cum_prediction

def make_batch_prediction(data_list):
    """Make batch prediction on data list"""

    predictions = []

    for d in data_list:
        prediction = make_prediction(d)
        predictions.append(prediction)

    return predictions

if __name__ == '__main__':
    predictions = make_batch_prediction([pd.read_csv('././data/test/control_2.csv'), 
    pd.read_csv('././data/test/control_8.csv'),
    pd.read_csv('././data/test/control_16.csv'),
    pd.read_csv('././data/test/control_20.csv'),
    pd.read_csv('././data/test/control_22.csv'),
    pd.read_csv('././data/test/condition_6.csv'),
    pd.read_csv('././data/test/condition_10.csv'),
    pd.read_csv('././data/test/condition_21.csv')])
    print(predictions)
