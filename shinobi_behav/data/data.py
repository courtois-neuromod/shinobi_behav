import numpy as np
import retro
import os.path as op
import pickle
import os
import json
from shinobi_behav.features.features import compute_max_score
from tqdm import tqdm
import time

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
    repetition_variables : dict
        A dict containing all the variables extracted from the log file
    """
    if setup == 'scan':
        level = filepath[-11:-8]
        timestring = filepath[-73:-58]
        timestamp = int(time.mktime(time.strptime(timestring,'%Y%m%d-%H%M%S')))

    if setup == 'home':
        level = filepath[-22]
        if level == '4':
            level = level + '-1'
        else:
            level = level + '-0'
        with open(filepath.replace('.bk2', '.json')) as json_file:
            metadata = json.load(json_file)
            timestamp = int(metadata['LevelStartTimestamp'])

    if level == '1-0':
        env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level1')
    elif level == '4-1':
        env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level4')
    elif level == '5-0':
        env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level5')
    actions = env.buttons

    repetition_variables = {}
    repetition_variables['filename'] = filepath
    repetition_variables['level'] = level
    repetition_variables['timestamp'] = timestamp

    key_log = retro.Movie(filepath)
    env.reset()

    while key_log.step():
        a = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
        _,_,done,i = env.step(a)

        for variable in i.keys(): # fill up dict
            if variable not in repetition_variables: # init entry first
                repetition_variables[variable] = []
            repetition_variables[variable].append(i[variable])
        for idx_a, action in enumerate(actions):
            if action not in repetition_variables:
                repetition_variables[action] = []
            repetition_variables[action].append(a[idx_a])
    env.close()
    return repetition_variables


def get_levelreps(path_to_data, subject, level, setup='home', remove_fake_reps=True):
    """Generates a list of the repetition_variables dicts for all repetitions of a level.

    Parameters
    ----------
    path_to_data : str
        The path to the data/ folder. Default is ./data/ or as defined in shinobi_behav.params
    subject : str
        Subject number, starts with 0 (ex. 01)
    level : str
        Level to collect. Can be '1-0', '4-1' or '5-0'
    setup : str, optional
        Can be 'scan', for files acquired during scanning sessions
        or 'home', for files acquired at home during the training sessions.

    Returns
    -------
    level_variables : dict
        A dict containing all the variables extracted from the log file
    """
    if setup == 'home':
        subject_template = op.join(path_to_data, 'shinobi_beh', '{}')
        session_template = op.join(path_to_data, 'shinobi_beh', '{}', '{}', 'beh')
        file_template = op.join(path_to_data, 'shinobi_beh', '{}', '{}', 'beh', '{}')
    elif setup == 'scan':
        subject_template = op.join(path_to_data, 'shinobi', 'sourcedata', '{}')
        session_template = op.join(path_to_data, 'shinobi', 'sourcedata', '{}', '{}')
        file_template = op.join(path_to_data, 'shinobi', 'sourcedata', '{}', '{}', '{}')


    n_fakereps = 0
    level_variables = []
    sessions  = [sesname for sesname in os.listdir(subject_template.format(subject)) if 'ses' in sesname]
    for sess in sorted(sessions):
        allfiles = [filename for filename in os.listdir(session_template.format(subject, sess)) if 'bk2' in filename]
        print('Processing : {} {}'.format(subject, sess))
        for file in tqdm(sorted(allfiles)):
            fpath = file_template.format(subject, sess, file)
            try:
                repetition_variables = extract_variables(fpath, setup=setup)

                if remove_fake_reps:# remove fake reps
                    if compute_max_score(repetition_variables) > 200:
                        level_variables.append(repetition_variables)
                    else:
                        n_fakereps += 1
                else:
                    level_variables.append(repetition_variables)
            except RuntimeError as e:
                print('Failed extraction for {} because of RuntimeError : '.format(file))
                print(e)
                continue
    if remove_fake_reps:
        print('Removed a total of {} fake reps (max score <= 200)'.format(n_fakereps))
    return level_variables, n_fakereps
