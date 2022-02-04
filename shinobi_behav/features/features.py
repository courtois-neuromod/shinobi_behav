import numpy as np
import pickle
import os.path as op
from datetime import datetime
from math import ceil
from scipy.stats import percentileofscore
import scipy.signal as signal


def load_features_dict(path_to_data, subject, level, setup, save=True, metric=None, days_of_train=True, allvars=None):
    """Load the features dict, create it if doesn't exists already.

    Parameters
    ----------
    path_to_data : str
        The path to the data/ folder. Default is ./data/ or as defined in shinobi_behav.params
    subject : str
        Subject number, starts with 0 (ex. sub-01)
    level : str
        Level. Can be '1-0', '4-1' or '5-0'
    setup : str, optional
        Can be 'scan', for files acquired during scanning sessions
        or 'home', for files acquired at home during the training sessions.
    save : bool, optional
        If True, will save the features dict if it needs to be created.
    metric : str, optional
        Can be 'mean' or 'median'. If metric is not None, the repetitions will
        be transformed with a moving average or moving median (n=10, step=1 by default)
    days_of_train : bool, optional
        If True, features will be expressed in number of days elapsed
        since the training started (broke ?)
    allvars : list, optional
        If the features dict doesn't exist yet, passes the allvars list to create it (WiP)

    Returns
    -------
    features_dict : dict
        A dict containing all the features computed from the given repetitions
    """

    features_dict_path = op.join(path_to_data, 'processed', f'{subject}_{level}_{setup}_repfeats_{metric}.pkl')
    if not(op.isfile(features_dict_path)):

        level_variables_path = op.join(path_to_data, 'processed',f'{subject}_{level}_allvars_{setup}.pkl')
        with open(level_variables_path, 'rb') as f:
            allvars = pickle.load(f)
        features_dict = aggregate_vars(allvars, metric=metric,
                           rel_speed=True,
                           health_lost=True,
                           max_score=True,
                           completion_prob=True,
                           completion_perc=True,
                           days_of_train=days_of_train)
        if save == True:
            with open(features_dict_path, 'wb') as f:
                pickle.dump(features_dict, f)
    else:
        with open(features_dict_path, 'rb') as f:
            features_dict = pickle.load(f)
    return features_dict

def aggregate_vars(allvars, metric=None, days_of_train=True, rel_speed=False, health_lost=False, max_score=False, completion_prob=False, completion_perc=False, completion_speed=False):
    '''
    Aggregate variables into repetition-level features and store them in a dict
    '''

    features_dict = {}
    start_of_training = allvars[0]['timestamp']
    for repvars in allvars:
        if days_of_train:
            features_dict['Days of training'] = compute_days_of_train(repvars, timestamp)
            print('Days of training computed')
        else:
            features_dict['Passage order'] =  [x for x in range(len(allvars['filename']))]
            print('Passage order computed')
        if rel_speed:
            features_dict['Relative speed'] = compute_rel_speed(allvars)
            print('Relative speed computed')
        if health_lost:
            features_dict['Health loss'] = compute_health_lost(allvars)
            print('Health loss computed')
        if max_score:
            features_dict['Max score'] = compute_max_score(allvars)
            print('Max score computed')
        if completion_prob:
            features_dict['Completion prob'] = compute_completed(allvars)
            print('Completion probability computed')
        if completion_perc:
            features_dict['Percent complete'] = compute_completed_perc(allvars)
            print('Completion percentage computed')
        if completion_speed:
            features_dict['Completion speed'] = compute_time2complete(allvars)
            print('Completion speed computed')

        if metric != None:
            for key in features_dict.keys():
                features_dict[key] = moving_descriptive(features_dict[key], N=10, metric=metric)
    return features_dict

def compute_days_of_train(repvars, timestamp):
    '''
    Translate timecodes into days-past-start-training. Starts at 1 instead of 0 (for exp/inverse fit)

    Inputs :
    allvars = dict with one entry per variable. Each entry contains a list of lists, with
    level1 lists = reps and level2 lists = frames

    Outputs :
    days_of_training = list with one element per repetition
    '''
    days_of_training= []
    first_day = []
    for timestamp in allvars['timestamp']:
        current_day = datetime.fromtimestamp(int(timestamp))
        if days_of_training == []:
            first_day = current_day
        d_training = current_day - first_day
        days_of_training.append(d_training.days+1)
    return days_of_training

def compute_max_score(repvars):
    """Get the maximum score of a repetition.

    Parameters
    ----------
    repvars : dict
        A dict containing all the variables extracted from the log file

    Returns
    -------
    max_score : int
        The maximum score reached across the repetition
    """
    max_score = max(repvars['score'])
    return max_score

def compute_health_lost(allvars):
    '''
    Total amount of health lost in each game.

    Inputs :
    allvars = dict with one entry per variable. Each entry contains a list of lists, with
    level1 lists = reps and level2 lists = frames

    Outputs :
    health_lost = list with one element per repetition
    '''
    health_lost = []
    for i in range(len(allvars['health'])):
        health_lost.append(sum([x for x in np.diff(allvars['health'][i], n=1) if x<0]))
    return health_lost


def compute_completed(allvars):
    '''
    Here we use "ended the level without losing a life" as a proxy for completed levels

    Inputs :
    allvars = dict with one entry per variable. Each entry contains a list of lists, with
    level1 lists = reps and level2 listprints = frames

    Outputs :
    completed = list with one element per repetition (0 for repetition failed, 1 for repetition completed)
    '''
    completed = []
    for repetition_lives in allvars['lives']:
        lives_lost = sum([x for x in np.diff(repetition_lives, n=1) if x<0])
        if lives_lost == 0:
            completed.append(1)
        else:
            completed.append(0)
    return completed

def compute_completed_perc(allvars):
    '''
    Here we use "reach somewhere around the end of the level" as a proxy for complete

    Inputs :
    allvars = dict with one entry per variable. Each entry contains a list of lists, with
    level1 lists = reps and level2 lists = frames

    Outputs :
    completed = list with one element per repetition (min 0 max 100)
    '''
    # Clean the X_player variable
    X_player = fix_position_resets(allvars['X_player'])

    # Find max value of X_player across all repetitions (that means that we assume that the level was completed at least once in our data)
    max_X = []
    for rep in X_player:
        max_X.append(np.max(rep))

    end_of_level = np.max(max_X) - 100 # lets assume that the level is completed when the player reaches end_of_level - 100, because there is some variance here due to the boss fights

    completed_perc = []
    # Now store the end position of each repetition and write it as percent of end_of_level
    for curr_X in max_X:
        if curr_X > end_of_level:
            curr_X = end_of_level
        completed_perc.append(curr_X/end_of_level*100)
    return completed_perc

def compute_time2complete(allvars):
    '''
    Number of frames elapsed until the last position in the level, only for repetitions that are completed.
    Failed repetitions are replaced by the average value of completed repetitions.

    Inputs :
    allvars = dict with one entry per variable. Each entry contains a list of lists, with
    level1 lists = reps and level2 lists = frames

    Outputs :
    time2complete = list with one element per repetition (number of frames until end)

    '''
    time2pos_lists = compute_time2pos(fix_position_resets(allvars['X_player']))
    completed = compute_completed(allvars)
    time2complete = []
    for i, r in enumerate(completed):
        if r:
            time2complete.append(time2pos_lists[i][-1])
        else:
            time2complete.append(np.nan)
    time2complete = [np.nanmean(time2complete) if np.isnan(x) else x for x in time2complete]
    return time2complete

def compute_rel_speed(allvars):
    '''
    Compute the average (per repetition) relative speed based on the distribution of time2pos, i.e. the number
    of frames elapsed until a position is reached.

    Inputs :
    allvars = dict with one entry per variable. Each entry contains a list of lists, with
    level1 lists = reps and level2 lists = frames

    Outputs :
    rel_speed = list with one element per repetition
    '''
    X_player_lists = fix_position_resets(allvars['X_player'])
    time2pos_lists = compute_time2pos(X_player_lists)
    distrib_t2p = distributions_t2p(time2pos_lists)
    rel_speed = []
    for i, run in enumerate(time2pos_lists):
        rel_speed.append(np.mean(compare_to_distrib(distrib_t2p, run)))
    return rel_speed

# relative speed prerequisites
def fix_position_resets(X_player_lists):
    '''
    Sometimes X_player resets to a previous value, but it's truly a new position.
    This fixes it and makes sure that X_player is continuous

    X_player_lists : A list of X_player arrays (one for each repetition)

    fixed_X_player_lists : same as X_player_lists
    '''
    fixed_X_player_lists = []
    for X_player in X_player_lists:
        fixed_X_player_list = []
        fix = 0
        for i in range(1, len(X_player)-1):
            if X_player[i-1] - X_player[i] > 100:
                fix += X_player[i-1] - X_player[i]
            fixed_X_player_list.append(X_player[i] + fix)
        fixed_X_player_list = fixed_X_player_list
        if fixed_X_player_list == []:
            fixed_X_player_list.append(32) # in case the list becomes empty add 32 so it is removed
        fixed_X_player_lists.append(fixed_X_player_list)
    return fixed_X_player_lists

def time2pos(run):
    '''
    Compute the number of frames to reach each position (i.e. value of X_player) in one singular run
    '''
    uniques = list(set(run)) # get unique values in list
    time2pos_run = []
    for val in uniques:
        time2pos_run.append(run.index(val)) # find first occurence of val
    return time2pos_run, uniques

def interpolate_missing_pos(time2pos_run, uniques):
    '''
    Some pos are missing, interpolate them by averaging the two nearest pos
    '''
    try:
        start = min(uniques)
        stop = max(uniques)
    except:
        start = 0
        stop = 0
    time2pos_full = []
    for pos in range(start, stop):
        if pos in uniques:
            time2pos_full.append(time2pos_run[uniques.index(pos)]) # append the value if pos in uniques
            last_frame = pos
        else:
            next_found = False
            next_frame = 1
            while next_found == False:
                if pos+next_frame in uniques:
                    interp_val = ceil((time2pos_run[uniques.index(last_frame)]+time2pos_run[uniques.index(pos+next_frame)])/2)
                    time2pos_full.append(interp_val)
                    next_found = True
                else:
                    next_frame = next_frame+1
    return time2pos_full

def sanitize_time2pos(time2pos_full):
    '''
    The player sometimes goes back in the level, and a pos previously reached
    but unregistered is reached again, registered this time.
    Get rid of these and cross your fingers that its not 2 in a row
    (should not happen if sfreq is high enough that the player cannot
    jump more than 2 pos, which seems to be the case)
    '''
    for i, pos in enumerate(time2pos_full):
        if pos < time2pos_full[i-1]:
            time2pos_full[i-1] = pos # if that happens, just replace the jumped pos by the next pos (=pos)
    time2pos_clean = time2pos_full
    return time2pos_clean

def compute_time2pos(X_player_lists):
    '''
    Transforms the list of positions (X_player variable) can be found in
    into the number of frames (time) to reach each pos.

    X_player_lists must first be corrected by fix_position_resets.
    '''
    time2pos_lists = []
    for run in X_player_lists:
        time2pos_run, uniques = time2pos(run)
        time2pos_full = interpolate_missing_pos(time2pos_run, uniques)
        time2pos_clean = sanitize_time2pos(time2pos_full)
        time2pos_lists.append(time2pos_clean[:-2])# no idea why but the last 2 values are 0 <<---- CHECK WHY
    return time2pos_lists

def distributions_t2p(time2pos_lists):
    '''
    Create distribution of the variable for each value
    '''
    distrib_t2p = []
    for i in range(max([len(time2pos) for time2pos in time2pos_lists])):
        pos_distrib = []
        for run in time2pos_lists:
            try:
                pos_distrib.append(run[i])
            except:
                pass
        distrib_t2p.append(np.array(pos_distrib))
    return distrib_t2p

def compare_to_distrib(distrib_t2p, time2pos_run):
    '''
    Compare an individual run to the distribution of all runs and get percentile for each pos.
    i.e. compute relative time_to_position
    '''
    pos_percentile = []
    for i, pos in enumerate(time2pos_run):
        pos_percentile.append(percentileofscore(distrib_t2p[i], pos))
    return pos_percentile


# aps computation
def compute_framewise_aps(repvars, actions, FS=60):
    # generate events for each of them
    action_mat = []
    for act in actions:
        var = repvars[act]
        var_bin = [int(val) for val in var]
        diffs = list(np.diff(var_bin, n=1))
        absdiffs = [abs(x) for x in diffs]
        action_mat.append(absdiffs)
    action_mat = np.array(action_mat)

    # compute number of action at each frame
    act_per_frame = np.zeros(action_mat.shape[1])
    for i in range(len(act_per_frame)):
        act_per_frame[i] = sum(action_mat[:,i])

    padded_apf = np.concatenate((np.zeros(int(FS/2)), act_per_frame, np.zeros(int(FS/2))))
    framewise_aps = np.zeros(len(padded_apf)-FS)
    for frame_idx in range(len(framewise_aps)):
        framewise_aps[frame_idx] = sum(padded_apf[frame_idx:frame_idx+FS])

    return framewise_aps


# utils
def filter_run(run, order=3, cutoff=0.005):
    '''
    Filter the relative time_to_position for visualization and computation of the derivative
    '''
    b, a = signal.butter(order,cutoff)
    run_filtered = signal.filtfilt(b, a, run)
    return run_filtered

def moving_descriptive(x, N=3, metric='mean'):
    x = np.array(x)
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    if metric == 'mean':
        out = list(np.mean(x[idx],axis=1))
    if metric == 'median':
        out = list(np.median(x[idx],axis=1))
    if metric == 'std':
        out = list(np.std(x[idx],axis=1))
    return out
