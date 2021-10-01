import numpy as np
import retro
import os.path as op
import pickle
import os
import json
from shinobi_behav.features.features import aggregate_vars, compute_max_score
from tqdm import tqdm

def retrieve_variables(files, level, bids=True, by_timestamps=True):
    '''
    TODO : clean the timestamps for non-bids files
    Retrieves all the variables from a list of bk2 files and creates the allvars structure.

    files : list of files with complete path
    level : the level corresponding to the savestate used
    bids : should be True for NUC sessions, False for scans

    allvars : dictionnary (each variable is an entry) containing list of arrays of
    length corresponding to the number of frames in each run,
    with runs ordered by timestamp.
    '''
    if level == '1':
        level = '1-0'
    if level == '4':
        level = '4-1'
    if level == '5':
        level = '5-0'

    if level == '1-0':
        env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level1')
    else:
        env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level'+level)
    actions = env.buttons
    variables_lists = {}

    # Extract timestamps
    if bids:
        timestamps = []
        for file in files:
            with open(file.replace('.bk2', '.json')) as json_file:
                data = json.load(json_file)
                timestamps.append(int(data['LevelStartTimestamp']))
    else:
        timestamps = []
        for file in files:
            timestamps.append(file[-14:-4])

    if by_timestamps:
        sorted_idx = np.argsort(timestamps)
    else:
        sorted_idx = range(len(timestamps))

    for idx in tqdm(sorted_idx, desc='Processing NUC files...'):
        file = files[idx]
        #print(file)
        run_variables = {}
        key_log = retro.Movie(file)
        env.reset()
        run_completed = False
        while key_log.step():
            a = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
            _,_,done,i = env.step(a)

            if variables_lists == {}: # init final dict
                variables_lists['filename'] = []
                variables_lists['timestamp'] = []
                variables_lists['level'] = []
                for action in actions:
                    variables_lists[action] = []
                for variable in i.keys():
                    variables_lists[variable] = []

            if run_variables == {}: # init temp dict
                for variable in i.keys():
                    run_variables[variable] = []
                for action in actions:
                    run_variables[action] = []

            for variable in i.keys(): # fill up temp dict
                run_variables[variable].append(i[variable])
            for idx_a, action in enumerate(actions):
                run_variables[action].append(a[idx_a])

            if done == True:
                run_completed = True
        variables_lists['filename'].append(file)
        variables_lists['timestamp'].append(timestamps[idx])
        variables_lists['level'].append(level)

        for variable in run_variables.keys():
            variables_lists[variable].append(run_variables[variable])
    env.close()
    allvars = variables_lists
    return allvars

def extract_variables(filepath, setup='scan'):
    """Runs the logfile to generate a dict that saves all the variables indexed
    in the data.json file.

    Parameters
    ----------
    file : str
        The path to the .bk2 file location
    setup : str, optional
        Can be 'scan', for files acquired during scanning sessions
        or 'home', for files acquired at home during the training sessions.

    Returns
    -------
    repvars : dict
        A dict containing all the variables extracted from the log file
    """

    level = filepath[-11:-8]
    date = filepath[-73:-65]

    if level == '1-0':
        env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level1')
    elif level == '4-1':
        env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level4-1')
    elif level == '5-0':
        env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level5')
    actions = env.buttons

    repvars = {}
    repvars['filename'] = filepath
    repvars['level'] = level
    repvars['date'] = date

    key_log = retro.Movie(filepath)
    env.reset()

    while key_log.step():
        a = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
        _,_,done,i = env.step(a)

        for variable in i.keys(): # fill up dict
            if variable not in repvars: # init entry first
                repvars[variable] = []
            repvars[variable].append(i[variable])
        for idx_a, action in enumerate(actions):
            if action not in repvars:
                repvars[action] = []
            repvars[action].append(a[idx_a])
    env.close()
    return repvars

def retrieve_scanvariables(files):
    # This is a modified version of src.data.data.retrieve_variables, adapted to the naming of scan-related behavioural files
    '''
    files : list of files with complete path

    variable_lists : dictionnary (each variable is an entry) containing list of arrays of
    length corresponding to the number of frames in each run,
    with runs ordered by timestamp.
    '''

    variables_lists = {}

    for file in tqdm(files, desc='Processing scan files...'):
        try:
            level = file[-11:-8]
            timestamp = file[-73:-65]

            if level == '1-0':
                env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level1')
            else:
                env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level'+level)
            actions = env.buttons

            run_variables = {}
            key_log = retro.Movie(file)
            env.reset()
            run_completed = False
            while key_log.step():
                a = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
                _,_,done,i = env.step(a)

                if variables_lists == {}: # init final dict
                    variables_lists['filename'] = []
                    variables_lists['timestamp'] = []
                    variables_lists['level'] = []
                    for action in actions:
                        variables_lists[action] = []
                    for variable in i.keys():
                        variables_lists[variable] = []

                if run_variables == {}: # init temp dict
                    for variable in i.keys():
                        run_variables[variable] = []
                    for action in actions:
                        run_variables[action] = []

                for variable in i.keys(): # fill up temp dict
                    run_variables[variable].append(i[variable])
                for idx_a, action in enumerate(actions):
                    run_variables[action].append(a[idx_a])

                if done == True:
                    run_completed = True
            variables_lists['filename'].append(file)
            variables_lists['timestamp'].append(timestamp)
            variables_lists['level'].append(level)

            for variable in run_variables.keys():
                variables_lists[variable].append(run_variables[variable])
            env.close()
        except Exception as e:
            print(e)
            print('Missing bk2 file')
    return variables_lists


def combine_variables(path_to_data, subject, level, behav=True, save=True):
    '''
    Load the raw data dict, create it if doesn't exists already.

    Inputs :
    path_to_data = string, path to the main BIDS folder
    subject = string, subject name
    level = string, level, can be '1','4','5' or '1-0', '4-1', '5-0'
    save = boolean, save the output in a file

    Outputs :
    allvars = dict, keys are raw variables. Each entry contains a list of len() = n_repetitions_total, in which each element is a list of len() = n_frames
    '''

    if behav:
        # select appropriate files
        sessions = os.listdir(op.join(path_to_data, 'shinobi_beh', subject))
        files = []
        for sess in sessions:
            allfiles = os.listdir(op.join(path_to_data, 'shinobi_beh', subject, sess, 'beh'))
            for file in allfiles:
                if 'level-{}'.format(level) in file:
                    if 'bk2' in file:
                        files.append(op.join(path_to_data, 'shinobi_beh', subject, sess, 'beh', file))
        allvars_path = op.join(path_to_data, 'processed','{}_{}_allvars_behav.pkl'.format(subject, level))

        # retrieve variables for the selected files
        if not(op.isfile(allvars_path)):
            allvars = retrieve_variables(files, level, bids=True)
            if save == True:
                with open(allvars_path, 'wb') as f:
                    pickle.dump(allvars, f)
        else:
            with open(allvars_path, 'rb') as f:
                allvars = pickle.load(f)
    else:
        # select appropriate files
        sessions = os.listdir(op.join(path_to_data, 'shinobi', 'sourcedata', subject))
        files = []
        for sess in sorted(sessions):
            allfiles = os.listdir(op.join(path_to_data, 'shinobi', 'sourcedata', subject, sess))
            for file in sorted(allfiles):
                if 'Level{}'.format(level) in file:
                    if 'bk2' in file:
                        files.append(op.join(path_to_data, 'shinobi', 'sourcedata', subject, sess, file))
        allvars_path = op.join(path_to_data, 'processed','{}_{}_allvars_scan.pkl'.format(subject, level))

        # retrieve variables for the selected files
        if not(op.isfile(allvars_path)):
            allvars = retrieve_variables(files, level, bids=False, by_timestamps=False)
            if save == True:
                with open(allvars_path, 'wb') as f:
                    pickle.dump(allvars, f)
        else:
            with open(allvars_path, 'rb') as f:
                allvars = pickle.load(f)
    return allvars

def remove_fake_reps(allvars):
    '''
    clean allvars from "fake-runs" (i.e. runs stopped without any moves from the player)
    '''

    scores = compute_max_score(allvars)
    i = len(scores)-1
    for score in reversed(scores):
        if score <= 200:
            for key in allvars.keys():
                del allvars[key][i]
        i -= 1
    return allvars
