# make_features.py

import numpy as np
import pandas as pd

def is_n_zeros(vals, n):
    """Checks if n-array of values are equal to zero"""
    return (len(vals) == n) and (sum(vals) == 0)

def is_n_ones(vals, n):
    """Checks if n-array of values are equal to one"""
    return (len(vals) == n) and vals.all()

def to_hours(delta):
    """Returns the hours equivalent of a delta"""
    return delta.seconds/3600

def sleep_re_score(scores):
    """Perform re scoring on the sleep scores
    
    Parameters
    __________

    scores: np.ndarray

    Returns
    _______
    scores: np.ndarray
    """

    re_scored = []
    for idx, _ in enumerate(scores):
        left_4_is_zero = is_n_zeros(scores[idx-4: idx], 4) # 00001
        left_10_is_zero = is_n_zeros(scores[idx-10: idx], 10) #00000000001
        left_15_and_right_15_is_zero = is_n_zeros(scores[idx-15: idx], 15) and not is_n_zeros(scores[idx: idx+6], 6) and \
            is_n_zeros(scores[idx+6: idx+23], 15) # Six 1's in between i.e 0000000000000001111110000000000000
        left_20_and_right_20_is_zero = is_n_zeros(scores[idx-20: idx], 20) and not is_n_zeros(scores[idx: idx+10], 10) and \
            is_n_zeros(scores[idx+10: idx+61], 20) # Six 1's in between i.e 00000000000000000000111111111100000000000000000000

        if left_20_and_right_20_is_zero: 
            # Add 10 strings of zero
            scores[idx: idx+10] = 0
            re_scored.append(0)
        if left_15_and_right_15_is_zero: 
            # Add 6 strings of zero
            scores[idx: idx+6] = 0
            re_scored.append(0)
        elif left_4_is_zero or left_10_is_zero:
            # Set to 0
            re_scored.append(0)
        else:
            re_scored.append(scores[idx])
    return np.array(re_scored)

def cole_krikpe(activity_data: np.ndarray) -> np.ndarray:
    """Computes sleep scoring
    
    Parameters
    __________

    activity_data:  np.ndarray
        Activity count

    Returns
    _______
    scores: array
        np.ndarray
    """

    scores = []
    
    def sleep_score(idx: int, act: int) -> int:
        preceeding_4 = np.array(activity_data[idx - 4: idx])
        succeeding_2 = np.array(activity_data[idx + 1: idx + 3])
        if len(preceeding_4) == 4 and len(succeeding_2) == 2:
            S = 0.0033 * ((preceeding_4[0] * 1.06) + (preceeding_4[1] * 0.54) + (preceeding_4[2] * 0.58) + (preceeding_4[3] * 0.76) + \
                (act * 2.30) + (succeeding_2[0] * 0.74) + (succeeding_2[1] * 0.67))
            return S
        return np.NaN

    for idx, act in enumerate(activity_data):
        scores.append(sleep_score(idx, act))

    scores = np.array(scores)

    mask1 = scores < 1
    mask2 = scores >= 1
    scores[mask1] = 1
    scores[mask2] = 0

    scores = sleep_re_score(scores)
    return scores

def mark_sleep_interval(sleep_score: np.ndarray) -> np.ndarray:
    """Mark sleep interval based on the sleep score"""

    intervals = []
    status = 'End'

    for idx, _ in enumerate(sleep_score):
        right_10_is_one = is_n_ones(sleep_score[idx: idx+10], 10)
        right_10_is_zero = is_n_zeros(sleep_score[idx: idx+10], 10)

        if status == 'End' and right_10_is_one:
            status = 'Start'
            intervals.append(status)
        elif status == 'Start' and right_10_is_zero:
            status = 'End'
            intervals.append(status)
        else:
            intervals.append(np.NaN)

    return intervals

def get_sleep_period(data):
    start_status = data[data['sleep status'] == 'Start'].index
    end_status = data[data['sleep status'] == 'End'].index
    df = pd.DataFrame()
    for i, j in zip(start_status, end_status):
        start = data.loc[i, 'timestamp']
        end  = data.loc[j, 'timestamp']
        date = data.loc[i, 'date']
        hours_slept = to_hours(end - start)
        df = df.append([[start, end, date, hours_slept]])
    df.columns = ['start', 'end', 'date', 'hours slept']
    return df

def get_avg_daily_sleep_period(data):
    df = get_sleep_period(data)
    grp = df.groupby('date')
    return grp.agg({'hours slept': ['mean', 'min', 'max']}).reset_index()

def next_batch(cond_a, cond_b) -> np.array:
    """Returns the next array batch based on the start and end condition"""

    start_idx = np.where(cond_a)
    end_idx = np.where(cond_b)

    for i, j in zip(start_idx[0], end_idx[0]):
        yield i + 1, j

def zero_runs(arr: np.array):
    """Returns the count of contiguous zeros in an array"""

    iszero =  np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges.shape[0]

def a_mean(sleep_status: np.array, activity: np.array) -> np.array:
    """Returns the average activity score while the subject is in bed"""

    a_mean_arr = []

    for i, j in next_batch(sleep_status == 'Start', sleep_status == 'End'):
        mask =  activity[i: j] != np.NaN
        data = pd.to_numeric(activity[i: j][mask])
        a_mean_arr.append(np.mean(data))

    return np.array(a_mean_arr)

def a_median(sleep_status: np.array, activity: np.array) -> np.array:
    """Returns the median activity score while the subject is in bed"""

    a_median_arr = []
    for i, j in next_batch(sleep_status == 'Start', sleep_status == 'End'):
        mask =  activity[i: j] != np.NaN
        data = pd.to_numeric(activity[i: j][mask])
        a_median_arr.append(np.median(data))

    return np.array(a_median_arr)

def a_std(sleep_status: np.array, activity: np.array) -> np.array:
    """Returns the standard deviation activity score while the subject is in bed"""

    a_std_arr = []
    for i, j in next_batch(sleep_status == 'Start', sleep_status == 'End'):
        mask =  activity[i: j] != np.NaN
        data = pd.to_numeric(activity[i: j][mask])
        a_std_arr.append(np.std(data))

    return np.array(a_std_arr)

def sleep_efficiency(sleep_status: np.array, sleep_score: np.array) -> np.array:
    """Returns the sleep efficiency of a subject"""

    se = []
    for i, j in next_batch(sleep_status == 'Start', sleep_status == 'End'):
        mask =  sleep_score[i: j] != 'nan'
        data = pd.to_numeric(sleep_score[i: j][mask])
        se.append(np.sum(data) / len(data) * 100)

    return np.array(se)

def tsmin(sleep_status: np.array, sleep_score: np.array) -> np.array:
    """Returns the number of minutes a subject was asleep between sleep onset and sleep offset (O-O interval)"""

    tsmin = []
    for i, j in next_batch(sleep_status == 'Start', sleep_status == 'End'):
        mask =  sleep_score[i: j] != 'nan'
        data = pd.to_numeric(sleep_score[i: j][mask])
        tsmin.append(np.sum(data))

    return np.array(tsmin)

def waso(sleep_status: np.array, sleep_score: np.array) -> np.array:
    """Returns the number of minutes a subject was awake between sleep onset and sleep offset (O–O interval)"""
    waso = []

    for i, j in next_batch(sleep_status == 'Start', sleep_status == 'End'):
        mask =  sleep_score[i: j] != 'nan'
        data = pd.to_numeric(sleep_score[i: j][mask])
        waso.append(len(data) - np.sum(data))

    return np.array(waso)

def wake_episode(sleep_status: np.array, sleep_score: np.array) -> np.array:
    """Returns the number of awakenings of a subject between sleep onset and sleep offset (O-O interval)"""

    we = []
    for i, j in next_batch(sleep_status == 'Start', sleep_status == 'End'):
        mask =  sleep_score[i: j] != 'nan'
        data = pd.to_numeric(sleep_score[i: j][mask])
        we.append(zero_runs(data))
    
    return we

def activity_index(sleep_status: np.array, activity: np.array) -> np.array:
    """Returns the proportion of minutes during TIB where the activity score was greater than zero."""

    a_index = []
    for i, j in next_batch(sleep_status == 'Start', sleep_status == 'End'):
        print(activity[i: j])
        mask =  activity[i: j] != np.NaN
        data = pd.to_numeric(activity[i: j][mask])

        num = len(data > 0)
        denum = len(data)
        a_index.append(num / denum)

    return a_index

if __name__ == "__main__":
    data = pd.read_csv('data/interim/condition/condition_1.csv', parse_dates=['timestamp'])
    sleep_score = cole_krikpe(data.activity)
    sleep_interval = mark_sleep_interval(sleep_score)
    print(activity_index(sleep_interval, data.activity))