import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.features.features import filter_run


def generate_key_events(var, key, FS=60):
    '''
    var = action variable, directly from allvars dict
    '''
    # always keep the first and last value as 0 so diff will register the state transition
    var[0] = 0
    var[-1] = 0

    var_bin = [int(val) for val in var]
    diffs = list(np.diff(var_bin, n=1))
    presses = [round(i/FS, 3) for i, x in enumerate(diffs) if x == 1]
    releases = [round(i/FS, 3) for i, x in enumerate(diffs) if x == -1]
    onset = presses
    duration = [round(releases[i] - presses[i], 3) for i in range(len(presses))]
    trial_type = ['{}'.format(key) for i in range(len(presses))]
    events_df = pd.DataFrame(data={'onset':onset,
                                   'duration':duration,
                                   'trial_type':trial_type})
    return events_df

def generate_aps_events(framewise_aps, FS=60, min_dur=1):
    filtered_aps = filter_run(framewise_aps, order=3, cutoff=0.002)
    var = filtered_aps

    median = np.median(var)

    mask_high = np.zeros(len(var))
    mask_low = np.zeros(len(var))

    for i, timestep in enumerate(var[1:-1]): # always keep the first and last value as 0 so diff will register the state transition
        if timestep < median:
            mask_low[i+1] = 1
        if timestep > median:
            mask_high[i+1] = 1

    diff_high = np.diff(mask_high, n=1)
    diff_low = np.diff(mask_low, n=1)

    durations_high = np.array([i for i, x in enumerate(diff_high) if x == -1]) - np.array([i for i, x in enumerate(diff_high) if x == 1])
    durations_low = np.array([i for i, x in enumerate(diff_low) if x == -1]) - np.array([i for i, x in enumerate(diff_low) if x == 1])

    #build df
    onset = []
    duration = []
    trial_type = []
    for i, dur in enumerate(durations_high):
        if dur >= (min_dur*FS):
            onset.append(np.array([i for i, x in enumerate(diff_high) if x == 1])[i]/FS)
            duration.append(durations_high[i]/FS)
            trial_type.append('high_APS')
    for i, dur in enumerate(durations_low):
        if dur >= (min_dur*FS):
            onset.append(np.array([i for i, x in enumerate(diff_low) if x == 1])[i]/FS)
            duration.append(durations_low[i]/FS)
            trial_type.append('low_APS')

    events_df = pd.DataFrame(data={'onset':onset,
                                   'duration':duration,
                                   'trial_type':trial_type})
    return events_df

# visualization
def plot_bidsevents(merged_df):
    event_ends = []
    for i in range(len(merged_df['onset'])):
        event_ends.append(merged_df['onset'][i] + merged_df['duration'][i])

    total_duration = max(event_ends)
    time_axis = np.linspace(0, total_duration, 10000)


    dict_to_plot = {}
    for ev_type in merged_df['trial_type'].unique():
        dict_to_plot[ev_type] = np.zeros(len(time_axis))

    for idx, line in merged_df.iterrows():
        for i, timepoint in enumerate(time_axis):
            if timepoint >= line['onset'] and timepoint <= line['onset']+line['duration']:
                dict_to_plot[line['trial_type']][i] = 1

    fig = plt.plot()
    plt.figure(figsize=(15,10))
    for i, key in enumerate(dict_to_plot.keys()):
        x = time_axis
        y = dict_to_plot[key]*(len(dict_to_plot.keys())-i)
        # remove 0's from plot
        for idx_val in reversed(range(len(y))):
            if y[idx_val] == 0:
                x = np.delete(x, idx_val)
                y = np.delete(y, idx_val)
        plt.scatter(x, y, label=key, marker = ',')
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    return fig
