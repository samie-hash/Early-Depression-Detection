# train.py

import datetime
import pickle
import sys
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join('.', 'src')))
from data.make_dataset import sleep_params_dataset

RAW_PATH = '././data/raw'
PROCESSED_PATH = '././data/processed'

def main():
    sleep_params_dataset(RAW_PATH, PROCESSED_PATH)

    # Read the data
    CONDITION_PROCESSED_FILE = '././data/processed/conditions002.csv'
    CONTROL_PROCESSED_FILE = '././data/processed/controls002.csv'

    conditions = pd.read_csv(CONDITION_PROCESSED_FILE)
    controls = pd.read_csv(CONTROL_PROCESSED_FILE)

    # Combine the data
    data = conditions.append(controls, ignore_index=True)

    # Shuffle the data
    data = data.sample(frac=1)

    # Split the data to train and test set
    X = data.drop('state', axis=1)
    y = data.state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=30)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # initialize the model
    xgb_clf = XGBClassifier(n_estimators=100, max_depth=2, use_label_encoder=False)
    xgb_clf.fit(X_train_scaled, y_train)
    print(f'Training Score: {xgb_clf.score(X_train_scaled, y_train)}')
    print(f'Testing Score: {xgb_clf.score(X_test_scaled, y_test)}')

    time_now = str(datetime.datetime.now())
    model_path = os.getcwd() + '/src/models/' +  'model_' + time_now.replace(':', '_') + '.pkl'
    pickle.dump(xgb_clf, open(model_path, 'wb'))

if __name__ == '__main__':
    main()