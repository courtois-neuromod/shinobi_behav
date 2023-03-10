import os.path as op
import os
import pickle
import json
import time
import shinobi_behav
import retro
#from shinobi_behav.features.features import compute_max_score
from shinobi_behav.utils import list_save
from tqdm import tqdm
from bids_loader.stimuli.game import get_variables_from_replay


def extract_variables(filepath, setup):
    """Runs the logfile to generate a dict that saves all the variables indexed
    in the data.json file.

    Parameters
    ----------
    filepath : str
        The path to the .bk2 file location
    setup : str
        Can be 'scan', for files acquired during scanning sessions
        or 'home', for files acquired at home during the training sessions.

    Returns
    -------
    repetition_variables : dict
        A Python dict with one entry per variable. In each entry, a list
        containing the state of a variable at each frame across a whole
        repetition.
    """
    if setup == "scan":
        level = filepath[-11:-8]
        timestring = filepath[-73:-58]
        timestamp = int(time.mktime(time.strptime(timestring, "%Y%m%d-%H%M%S")))

    if setup == "home":
        level = filepath[-22]
        with open(filepath.replace(".bk2", ".json")) as json_file:
            metadata = json.load(json_file)
            timestamp = int(metadata["LevelStartTimestamp"])

    env = retro.make("ShinobiIIIReturnOfTheNinjaMaster-Genesis", state=f"Level{level}")

    key_log = retro.Movie(filepath)
    env.reset()
    repetition_variables = init_variables_dict(filepath, level, timestamp, env, key_log)
    # Starts accumulating variables states
    env.reset()
    while key_log.step():
        action_list = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
        _, _, _, frame_variables = env.step(action_list)
        repetition_variables = store_frame_variables(
            repetition_variables, frame_variables, action_list, env
        )
    env.close()
    return repetition_variables


def init_variables_dict(filepath, level, timestamp, env, key_log):
    """Create and initialize the dict that will contain the variables states of
    a repetition.

    Parameters
    ----------
    filepath : str
        The path to the .bk2 file location
    level : type
        The level at which the repetition was played. Can be found in filename.
    timestamp : int
        The UNIX timestamp associated with the repetition's beginning.
    env : retro Env
        The current env, to obtain variables names.

    Returns
    -------
    dict
        A Python dict with one empty entry per variable.

    """
    repetition_variables = {}
    repetition_variables["filename"] = filepath
    repetition_variables["level"] = level
    repetition_variables["timestamp"] = timestamp

    # Run the first step to obtain variable keys
    action_list = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
    _, _, _, frame_variables = env.step(action_list)

    # Init all entries
    for key in frame_variables:
        repetition_variables[key] = []
    for action in env.buttons:
        repetition_variables[action] = []
    return repetition_variables


def store_frame_variables(repetition_variables, frame_variables, action_list, env):
    """Append the variables states of a frame to the repetition dict.

    Parameters
    ----------
    repetition_variables : dict
        A Python dict with one entry per variable. In each entry, a list
        containing the state of a variable at each frame.
    frame_variables : dict
        A Python dict containing the variables states at the current frame.
    action_list : list
        List of player inputs states at the current frame. Represented by a 0
        or 1 for each action.
    env : type
        The current env, to obtain actions names.

    Returns
    -------
    dict
        A Python dict with one entry per variable.

    """
    for key, item in frame_variables.items():
        repetition_variables[key].append(item)
    for idx_action, action in enumerate(env.buttons):
        repetition_variables[action].append(action_list[idx_action])
    return repetition_variables


def get_levelreps(path_to_data, subject, level, setup, remove_fake_reps=True):
    """Generates a list of the repetition_variables dicts for all repetitions
     of a level.

    Parameters
    ----------
    path_to_data : str
        The path to the data/ folder.
        Default is ./data/ or as defined in shinobi_behav.params
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
    levelwise_variables : list
        A list of dicts containing all the variables extracted from the log
        file of all the repetitions of a subject in a specific level, on a
        specific setup.
    """
    if setup == "home":
        subject_template = op.join(path_to_data, "shinobi_training", "{}")
        session_template = op.join(path_to_data, "shinobi_training", "{}", "{}", "beh")
        file_template = op.join(
            path_to_data, "shinobi_training", "{}", "{}", "beh", "{}"
        )
        skip_first_step = False
    elif setup == "scan":
        subject_template = op.join(path_to_data, "shinobi", "{}")
        session_template = op.join(path_to_data, "shinobi", "{}", "{}", "gamelogs")
        file_template = op.join(path_to_data, "shinobi", "{}", "{}", "gamelogs", "{}")

    names_fakereps = []
    names_emptyfiles = []
    levelwise_variables = []
    sessions = [
        sesname
        for sesname in os.listdir(subject_template.format(subject))
        if "ses" in sesname
    ]
    for sess in sorted(sessions):
        allfiles = [
            filename
            for filename in os.listdir(session_template.format(subject, sess))
            if "bk2" in filename
            and level in filename
        ]
        if setup == "scan":
            runlist = [x.split("_")[3] for x in sorted(allfiles)]
            skip_first_steps = [True if runlist[i-1] != x else False for i, x in enumerate(runlist)]
        elif setup == "home":
            skip_first_steps = [True for x in allfiles]
        print("Processing : {} {}".format(subject, sess))
        for idx_file, file in tqdm(enumerate(sorted(allfiles))):
            fpath = file_template.format(subject, sess, file)
            try:
                #repetition_variables = extract_variables(fpath, setup=setup)
                repetition_variables = get_variables_from_replay(fpath, skip_first_step=skip_first_steps[idx_file],
                                                                 inttype=retro.data.Integrations.STABLE)

                if remove_fake_reps:
                    if max(repetition_variables["score"]) > 200:
                        levelwise_variables.append(repetition_variables)
                    else:
                        names_fakereps.append(fpath)
                else:
                    levelwise_variables.append(repetition_variables)
            except RuntimeError as error:
                print(f"Failed extraction for {file} because of RuntimeError : ")
                print(error)
                names_emptyfiles.append(fpath)

    if remove_fake_reps:
        print(f"Removed a total of {len(names_fakereps)} fake reps (max score <= 200)")
    print(f"Found a total of {len(names_emptyfiles)} empty files")
    return levelwise_variables, names_fakereps, names_emptyfiles
