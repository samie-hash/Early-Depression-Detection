# visualize.py
import os
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
import numpy as np
import seaborn as sns

def plot_histogram(x: str, data: array=None, title: str="", filename: str=None, **kwargs) -> None:
    """Plots histogram of activity count
    
    x:  str
        Column name of the data to plot
    data:   array
        numpy.ndarray object
    title:  str
        Title of the plot
    filename: str
        Filename of the plot
    **kwargs: dict
        Key word arguments to pass to the plot constructor
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(x=x, data=data, ax=ax, **kwargs)
    plt.title(title, fontsize=14)
    if filename:
        fig.savefig(filename, dpi=82)

def plot_sleep_period(data, error_bar: bool=True, title:str="", filename: str=None) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    x = data.date
    y =  data.iloc[0:, 1]

    yerr = [data.iloc[0:, 1] - data.iloc[0:, 2], data.iloc[0:, 3] - data.iloc[0:, 1]]
    yerr = np.array(yerr)
    plt.errorbar(x,y, yerr=yerr, color='r', fmt='o-', mfc='blue', mec='green', ms=10, ecolor='green')
    plt.title(title, fontsize=16)
    plt.xticks(rotation=30)
    
    if filename:
        fig.savefig(filename, dpi=82)

def convert_to_categorical_time(x):
    if (x >= 0) & (x < 6):
        time_of_day = 'night'
    elif (x >= 6) & (x < 12):
        time_of_day = 'morning'
    elif (x >= 12) & (x < 18):
        time_of_day = 'afternoon'
    else:
        time_of_day = 'evening'
    return time_of_day

def select_random_data():
    "Randomly select a patient data from the control and condition path "
    # set up the directories
    control_path = '../data/raw/control/'
    condition_path = '../data/raw/condition/'

    control_files = os.listdir(control_path)
    condition_files = os.listdir(condition_path)

    control_filename = random.choice(control_files)
    condition_filename = random.choice(condition_files)

    control = pd.read_csv(control_path + control_filename, parse_dates=['timestamp'], index_col='timestamp')
    condition = pd.read_csv(condition_path + condition_filename, parse_dates=['timestamp'], index_col='timestamp')

    return control, control_filename, condition, condition_filename

def view_random_activity():
    """Randomly select a patient data from the control and condition path and plot a time series graph"""
    
    control, control_filename, condition, condition_filename  = select_random_data()

    fig, axes = plt.subplots(2, 1, figsize=(16, 6))
    fig.tight_layout(pad=4.0)

    axes[0].plot(control.index, control.activity)
    axes[0].set_title(f'Time series activity graph for {control_filename} patient (non-Depressed)', fontsize=16)

    axes[1].plot(condition.index, condition.activity)
    axes[1].set_title(f'Time series activity graph for {condition_filename} patient (depressed)', fontsize=16)
    plt.show()


def view_seasonality():
    """Randomly select a patient data from the control and condition path and plot a time series seasonality ('morning', 'afternoon', 'evening', 'night') graph"""

    control, control_filename, condition, condition_filename  = select_random_data()

    # get the time of the day from timestamp data ('morning', 'afternoon', 'evening', 'night')
    control['time_of_day'] = control.index.hour.map(lambda x: convert_to_categorical_time(x))
    control['day'] = control.index.day

    condition['time_of_day'] = condition.index.hour.map(lambda x: convert_to_categorical_time(x))
    condition['day'] = condition.index.day

    np.random.seed(100)

    fig, axes = plt.subplots(2, 1, figsize=(16,6))
    fig.tight_layout(pad=4.0)
    #plt.subplots_adjust(right=None, top=1.6)

    # control patient plot
    days = control['day'].unique()
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(days), replace=False)

    for i, d in enumerate(days):
        if i > 0:        
            df = control.loc[control.day==d, :]
            new_df = df.groupby(['time_of_day'])['activity'].median().reset_index()
            new_df['time_of_day'] = pd.Categorical(new_df['time_of_day'], categories=['morning', 'afternoon', 'evening', 'night'], ordered=True)
            new_df = new_df.sort_values('time_of_day')
            axes[0].plot('time_of_day', 'activity', data=new_df, color=mycolors[i], label=d)
            axes[0].set_title(f'Seasonal plot for {control_filename} patient (non-depressed) activity data')
            
    # condition patient plot
    days = condition['day'].unique()
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(days), replace=False)

    for i, d in enumerate(days):
        if i > 0:        
            df = condition.loc[condition.day==d, :]
            new_df = df.groupby(['time_of_day'])['activity'].median().reset_index()
            new_df['time_of_day'] = pd.Categorical(new_df['time_of_day'], categories=['morning', 'afternoon', 'evening', 'night'], ordered=True)
            new_df = new_df.sort_values('time_of_day')
            axes[1].plot('time_of_day', 'activity', data=new_df, color=mycolors[i], label=d)
            axes[1].set_title(f'Seasonal plot for {condition_filename} patient (depressed) activity data')

    return fig