import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shinobi_behav.features.features import filter_run, compute_framewise_aps
import matplotlib


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

def generate_healthloss_events(health, FS=60, dur=0.1):
    '''
    health : variable 'health' from the runvars structure
    FS : sampling frequency, 60 by default
    dur : arbitrary duration of the event

    returns events_df : a bids-like events dataframe

    '''
    diff_health = np.diff(health, n=1)

    onset = []
    duration = []
    trial_type = []
    for idx, x in enumerate(diff_health):
        if x < 0:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('HealthLoss')
        if x > 0:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('HealthGain')

    #build df
    events_df = pd.DataFrame(data={'onset':onset,
                               'duration':duration,
                               'trial_type':trial_type})
    return events_df

def create_runevents(runvars, startevents, actions, FS=60, min_dur=1, get_aps=True, get_actions=True, get_healthloss=True, get_startend=True):
    onset_reps = startevents['onset'].values.tolist()
    dur_reps = startevents['duration'].values.tolist()

    if get_aps:
        try:
            framewise_aps = compute_framewise_aps(runvars, actions=actions, FS=FS)
        except Exception as e:
            print(e)

    # init df list
    all_df = []

    for idx, onset_rep in enumerate(onset_reps):
        if isinstance(startevents['stim_file'][idx], str):
            lvl_rep =  startevents['stim_file'][idx][-11]

            #print('Extracting events for {}'.format(runvars['filename'][idx]))
            if get_actions:
                # get the different possible actions
                # generate events for each of them
                for act in actions:
                    var = runvars[act][idx]
                    temp_df = generate_key_events(var, act, FS=FS)
                    temp_df['onset'] = temp_df['onset'] + onset_rep
                    temp_df['trial_type'] = lvl_rep + '_' + temp_df['trial_type']
                    all_df.append(temp_df)
            if get_aps:
                temp_df = generate_aps_events(framewise_aps, FS=FS)
                temp_df['onset'] = temp_df['onset'] + onset_rep
                temp_df['trial_type'] = lvl_rep + '_' + temp_df['trial_type']
                all_df.append(temp_df)
            if get_healthloss:
                temp_df = generate_healthloss_events(runvars['health'][idx], FS=FS, dur=0.1)
                temp_df['onset'] = temp_df['onset'] + onset_rep
                temp_df['trial_type'] = lvl_rep + '_' + temp_df['trial_type']
                all_df.append(temp_df)

        if all_df != []:
            if get_startend:
                temp_df = startevents.drop('stim_file', axis=1)
                temp_df['trial_type'] = 'level_{}'.format(lvl_rep)
                all_df.append(temp_df)
                #todo : if get_endstart
                #todo : if get_kills
                #todo : if get_healthloss

            events_df = pd.concat(all_df).sort_values(by='onset').reset_index(drop=True)
        else:
            events_df = pd.DataFrame()
        return events_df


def trim_events_df(events_df, trim_by='LvR'):
    if trim_by=='LvR':
        # Create Left df
        lh_df = pd.concat([events_df[events_df['trial_type'] == '1_LEFT'],
                           events_df[events_df['trial_type'] == '1_RIGHT'],
                           events_df[events_df['trial_type'] == '1_DOWN'],
                           events_df[events_df['trial_type'] == '1_UP'],
                           events_df[events_df['trial_type'] == '4_LEFT'],
                           events_df[events_df['trial_type'] == '4_RIGHT'],
                           events_df[events_df['trial_type'] == '4_DOWN'],
                           events_df[events_df['trial_type'] == '4_UP'],
                            events_df[events_df['trial_type'] == '5_LEFT'],
                           events_df[events_df['trial_type'] == '5_RIGHT'],
                           events_df[events_df['trial_type'] == '5_DOWN'],
                           events_df[events_df['trial_type'] == '5_UP']
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_df['trial_type'] = 'LeftH'

        # Create Right df
        rh_df = pd.concat([events_df[events_df['trial_type'] == '1_B'],
                           events_df[events_df['trial_type'] == '1_C'],
                           events_df[events_df['trial_type'] == '4_B'],
                           events_df[events_df['trial_type'] == '4_C'],
                            events_df[events_df['trial_type'] == '5_B'],
                           events_df[events_df['trial_type'] == '5_C']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_df['trial_type'] = 'RightH'
        # Regroup and pass them
        trimmed_df = pd.concat([lh_df, rh_df]).sort_values(by='onset').reset_index(drop=True)

    if trim_by=='event':
        # mostly for plotting
        lh_l = pd.concat([ events_df[events_df['trial_type'] == '1_LEFT'],
                           events_df[events_df['trial_type'] == '4_LEFT'],
                           events_df[events_df['trial_type'] == '5_LEFT']
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_l['trial_type'] = 'Left hand - Move left'
        lh_r = pd.concat([ events_df[events_df['trial_type'] == '1_RIGHT'],
                           events_df[events_df['trial_type'] == '4_RIGHT'],
                           events_df[events_df['trial_type'] == '5_RIGHT']
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_r['trial_type'] = 'Left hand - Move right'
        lh_d = pd.concat([ events_df[events_df['trial_type'] == '1_DOWN'],
                           events_df[events_df['trial_type'] == '4_DOWN'],
                           events_df[events_df['trial_type'] == '5_DOWN'],
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_d['trial_type'] = 'Left hand - Move down'
        lh_u = pd.concat([ events_df[events_df['trial_type'] == '1_UP'],
                           events_df[events_df['trial_type'] == '4_UP'],
                           events_df[events_df['trial_type'] == '5_UP']
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_u['trial_type'] = 'Left hand - Move up'
        rh_jump = pd.concat([events_df[events_df['trial_type'] == '1_B'],
                           events_df[events_df['trial_type'] == '4_B'],
                            events_df[events_df['trial_type'] == '5_B']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_jump['trial_type'] = 'Right hand - Jump'
        rh_hit = pd.concat([events_df[events_df['trial_type'] == '1_C'],
                           events_df[events_df['trial_type'] == '4_C'],
                           events_df[events_df['trial_type'] == '5_C']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_hit['trial_type'] = 'Right hand - Hit'
        hl = pd.concat([events_df[events_df['trial_type'] == '1_HealthLoss'],
                           events_df[events_df['trial_type'] == '4_HealthLoss'],
                           events_df[events_df['trial_type'] == '5_HealthLoss']
                          ]).sort_values(by='onset').reset_index(drop=True)
        hl['trial_type'] = 'Health loss'
        trimmed_df = pd.concat([lh_l, lh_r, lh_u, lh_d, rh_jump, rh_hit, hl]).sort_values(by='onset').reset_index(drop=True)

    if trim_by=='healthloss':
        hl = pd.concat([events_df[events_df['trial_type'] == '1_HealthLoss'],
                           events_df[events_df['trial_type'] == '4_HealthLoss'],
                           events_df[events_df['trial_type'] == '5_HealthLoss']
                          ]).sort_values(by='onset').reset_index(drop=True)
        hl['trial_type'] = 'HealthLoss'
        trimmed_df = hl

    if trim_by=='JvH':
        # Create Left df
        rh_jump = pd.concat([events_df[events_df['trial_type'] == '1_B'],
                           events_df[events_df['trial_type'] == '4_B'],
                            events_df[events_df['trial_type'] == '5_B']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_jump['trial_type'] = 'Jump'
        rh_hit = pd.concat([events_df[events_df['trial_type'] == '1_C'],
                           events_df[events_df['trial_type'] == '4_C'],
                           events_df[events_df['trial_type'] == '5_C']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_hit['trial_type'] = 'Hit'
        trimmed_df = pd.concat([rh_jump, rh_hit]).sort_values(by='onset').reset_index(drop=True)

    return trimmed_df




##########
def plot_gameevents(events_df, colors='rand'):
    '''
     colors : can be 'rand, 'lvr' or specified by a list of 3-tuples with a length corresponding to the number of event_types
     if 'lvr' : Left vs Right hand, colors will be forced to match between different buttons of the same hand
    '''

    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=40)

    trial_types = sorted(list(events_df.trial_type.unique()))

    event_ends = []
    for i in range(len(events_df['onset'])):
        event_ends.append(events_df['onset'][i] + events_df['duration'][i])

    total_duration = max(event_ends)
    time_axis = np.linspace(0, total_duration, 10000)

    if colors=='lvr':
        cmap_right = matplotlib.cm.get_cmap('hot')
        cmap_left = matplotlib.cm.get_cmap('cool')
        col_bank = [(0,0,0),cmap_left(0.5),cmap_left(0.4),cmap_left(0.3),cmap_left(0.2),cmap_right(0.5),cmap_right(0.3)]
    elif colors =='rand': # Generate random colors
        col_bank = []
        for i in range(0, len(trial_types)):
            col_bank.append(tuple(np.random.choice(range(0, 10), size=3)/10))
    else:
         col_bank = colors

    masks = []
    colors_segs = []
    segs = []
    for line in LvR_df.T.iteritems():
        onset = line[1][0]
        dur = line[1][1]
        ttype = line[1][2]
        segs.append([(onset, trial_types.index(ttype)+1), (onset+dur, trial_types.index(ttype)+1)])
        mask = np.ma.masked_where((time_axis > onset) & (time_axis < onset+dur), time_axis)
        masks.append(mask)
        colors_segs.append(col_bank[trial_types.index(ttype)])

    # create figure
    lc = mc.LineCollection(segs, colors=colors_segs, linewidths=77)
    fig, ax = plt.subplots(figsize=(15,10))

    ax.add_collection(lc)
    ax.set_yticks(np.arange(len(trial_types))+1)
    ax.set_yticklabels(trial_types)
    ax.set_ylim((0.5,len(trial_types)+0.5))
    ax.set_xlim((0,max(time_axis)))
    ax.set_xlabel('Time (s)', fontsize=30)
    ax.margins(0.1)
    return fig, ax

################# DEPRECATED
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
