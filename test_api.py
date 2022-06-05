import requests, json
import pandas as pd
import numpy as np
# from src.data.make_dataset import process_data

data = pd.read_csv('data/test/condition_21.csv')
# preprocessed_data = process_data(data)
# final_data = json.dumps(preprocessed_data.to_dict())
url = 'http://127.0.0.1:2000/predict'

send_request=requests.post(url, data.to_dict())
print(send_request.json())