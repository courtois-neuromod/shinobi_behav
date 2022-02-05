import retro
import os.path as op
import pickle
import os
import json
from shinobi_behav.features.features import compute_max_score
from tqdm import tqdm
import time
import shinobi_behav


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
        action_list = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
        _, _, _, state_variables = env.step(action_list)

        for variable in state_variables.keys(): # fill up dict

            if variable not in repetition_variables: # init entry first
                repetition_variables[variable] = []
            repetition_variables[variable].append(state_variables[variable])
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
    remove_fake_reps : bool
        If True, ignore files with max_score < 200

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

    names_fakereps = []
    names_emptyfiles = []
    level_variables = []
    sessions  = [sesname for sesname in os.listdir(subject_template.format(subject)) if 'ses' in sesname]
    for sess in sorted(sessions):
        allfiles = [filename for filename in os.listdir(session_template.format(subject, sess)) if 'bk2' in filename]
        print('Processing : {} {}'.format(subject, sess))
        for file in tqdm(sorted(allfiles)):
            fpath = file_template.format(subject, sess, file)
            try:
                repetition_variables = extract_variables(fpath, setup=setup)

                if remove_fake_reps:
                    if compute_max_score(repetition_variables) > 200:
                        level_variables.append(repetition_variables)
                    else:
                        names_fakereps.append(fpath)
                else:
                    level_variables.append(repetition_variables)
            except RuntimeError as e:
                print(f'Failed extraction for {file} because of RuntimeError : ')
                print(e)
                names_emptyfiles.append(fpath)

    if remove_fake_reps:
        print(f'Removed a total of {len(names_fakereps)} fake reps (max score <= 200)')
    print(f'Found a total of {len(names_emptyfiles)} empty files (leading to "movie could not be loaded" errors)')
    return level_variables, names_fakereps, names_emptyfiles


def extract_and_save_corrupted_files():
    for setup in ['home', 'scan']:
        emptyfiles = []
        fakereps = []
        for subj in shinobi_behav.SUBJECTS:
            for level in ['1-0', '4-1', '5-0']:
                varfile_fpath = op.join(shinobi_behav.DATA_PATH, 'processed', f'{subj}_{level}_allvars_{setup}.pkl')
                with open(varfile_fpath, 'rb') as f:
                    _, names_fakereps, names_emptyfiles = pickle.load(f)
                fakereps.append(names_fakereps)
                emptyfiles.append(names_emptyfiles)
        emptyfiles_fname = op.join(shinobi_behav.DATA_PATH, 'processed', f'{setup}_emptyfiles.pkl')
        fakereps_fname = op.join(shinobi_behav.DATA_PATH, 'processed', f'{setup}_fakereps.pkl')
        pickle_save(emptyfiles_fname, emptyfiles)
        pickle_save(fakereps_fname, fakereps)


def pickle_save(fpath, content):
    '''
    Saves the content in a pickle file

    Parameters
    ----------
    fpath : str
        Path to the file to be created
    content : str
        Python object
    '''
    with open(fpath, 'wb') as f:
        pickle.dump(content, f)
