import numpy as np
import pickle
import os.path as op
from datetime import datetime
from math import ceil
from scipy.stats import percentileofscore
import scipy.signal as signal
from shinobi_behav.utils import pickle_save
from shinobi_behav.utils import moving_descriptive
import json

def load_features_dict(
    path_to_data, subject, level, setup, save=True, metric=None, days_of_train=True
):
    """Load the features dict, creates it from the processed variables files
    if doesn't exists already.

    Parameters
    ----------
    path_to_data : str
        The path to the data/ folder. Default is ./data/ or as defined in
        shinobi_behav.params
    subject : str
        Subject number, starts with sub-0 (ex. sub-01)
    level : str
        Level. Can be '1', '4' or '5'
    setup : str, optional
        Can be 'scan', for files acquired during scanning sessions
        or 'home', for files acquired at home during the training sessions.
    save : bool, optional
        If True, will save the features dict if it needs to be created.
    metric : str, optional
        Can be 'mean' or 'median'. If metric is not None, the repetitions will
        be transformed with a moving average or moving median
        (with n=10, step=1 by default).
    days_of_train : bool, optional
        If True, features will be expressed in number of days elapsed
        since the training started (broke ?)

    Returns
    -------
    features_dict : dict
        A dict containing all the features computed from the given repetitions
    """

    features_dict_path = op.join(
        path_to_data, "processed", "levelwise_variables", f"{subject}_{level}_{setup}_repfeats_{metric}.pkl"
    )
    if not op.isfile(features_dict_path):

        level_variables_path = op.join(
            path_to_data,
            "processed",
            "levelwise_variables",
            f"{subject}_{level}_levelwise_variables_{setup}.pkl",
        )
        with open(level_variables_path, "rb") as f:
            levelwise_variables = pickle.load(f)
        features_dict = compute_features(
            levelwise_variables,
            metric=metric,
            rel_speed=True,
            health_lost=True,
            max_score=True,
            completion_prob=True,
            completion_perc=True,
            days_of_train=days_of_train,
        )
        if save == True:
            pickle_save(features_dict_path, features_dict)
    else:
        with open(features_dict_path, "rb") as f:
            features_dict = pickle.load(f)
    return features_dict


def compute_features(
    list_variables,
    metric=None,
    days_of_train=True,
    rel_speed=False,
    health_lost=False,
    max_score=False,
    completion_prob=False,
    completion_perc=False,
    completion_speed=False,
):
    """Computes performance metrics based on the repetition_variables dict.
    Some of these metrics, like "rel speed", require that each repetition is
    compared against the distribution of all other repetitions.

    Parameters
    ----------
    list_variables : list
        Any list of repetition_variables dicts.
    metric : str
        If not none, will apply a summary metrics in a moving window
        (with n=10, step=1 by default). Can be "mean" or "median".
    days_of_train : bool
        If True, will index the datapoints by the number of days elapsed since
        the first game of the dataset was played. If False, the data are ordered
        by passage order.
    rel_speed : bool
        If True, computes the average relative speed across the repetitions.
        See features.compute_rel_speed for more information.
    health_lost : bool
        If True, computes the total amount of health lost during the repetitions.
    max_score : bool
        If True, computes the maximum score reached in the repetitions.
    completion_prob : bool
        If True, computes the probability of a repetition to be completed, given
        the moving window. Must be used with the metric argument set to something
        else than None, or else it will only contain binary values (0 or 1) for
        each repetition.
    completion_perc : bool
        If True, computes the proportion of the level that was completed during
        the repetitions. Expressed from 0 to 1.
    completion_speed : bool
        If True, computes the time (number of frames) elapsed until the
        completion of the repetition.

    Returns
    -------
    dict
        Dictionnary containing one entry per performance metrics, each containing
        List of length = n_repetitions.

    """

    features_dict = {}
    # for repetition_variables in list_variables:
    if days_of_train:
        features_dict["Days of training"] = compute_days_of_train(list_variables)
        print("Days of training computed")
    else:
        features_dict["Passage order"] = list(range(len(list_variables)))
        print("Passage order computed")
    if rel_speed:
        features_dict["Relative speed"] = compute_rel_speed(list_variables)
        print("Relative speed computed")
    if health_lost:
        features_dict["Health loss"] = compute_health_lost(list_variables)
        print("Health loss computed")
    if max_score:
        features_dict["Max score"] = compute_max_score(list_variables)
        print("Max score computed")
    if completion_prob:
        if metric is not None:
            features_dict["Completion probability"] = compute_completed(list_variables)
            print("Completion probability computed")
        else:
            features_dict["Completion"] = compute_completed(list_variables)
            print("Completion computed")
    if completion_perc:
        features_dict["Percent complete"] = compute_completed_perc(list_variables)
        print("Completion percentage computed")
    if completion_speed:
        features_dict["Completion speed"] = compute_time2complete(list_variables)
        print("Completion speed computed")

    if metric is not None:
        for key in features_dict:
            features_dict[key] = moving_descriptive(
                features_dict[key], win_size=10, metric=metric
            )
    return features_dict


def compute_days_of_train(levelwise_variables):
    """Translate timecodes into days-past-start-training.
    Starts at 1 instead of 0 (for exp/inverse fit).

    Parameters
    ----------
    levelwise_variables : list of dicts
        List of dicts containing all the variables extracted from the log
        file of all the repetitions of a subject in a specific level, on a
        specific setup.

    Returns
    -------
    list
        List of datetime objects indicating the number of days elapsed since
        the training's beginning, for each repetition.

    """
    days_of_training = []
    timestamps = []
    for repetition_dict in levelwise_variables:
        json_fname = repetition_dict['filename'].replace(".bk2", ".json")
        with open(json_fname, "r") as f:
            json_dict = json.load(f)
        timestamps.append(json_dict["LevelStartTimestamp"])
        
    #timestamps.append(repetition_dict["timestamp"])
    first_day = datetime.fromtimestamp(np.min(timestamps))

    for timestamp in timestamps:
        current_day = datetime.fromtimestamp(int(timestamp))
        d_training = current_day - first_day
        days_of_training.append(d_training.days + 1)
    return days_of_training


def compute_max_score(levelwise_variables):
    """Gets the maximum score attained at each repetition.

    Parameters
    ----------
    levelwise_variables : list of dicts
        List of dicts containing all the variables extracted from the log
        file of all the repetitions of a subject in a specific level, on a
        specific setup.

    Returns
    -------
    list
        List of maximum score values.

    """
    max_scores = []
    for repetition_dict in levelwise_variables:
        max_scores.append(max(repetition_dict["score"]))
    return max_scores


def compute_health_lost(levelwise_variables):
    """Total amount of health lost in each repetition.

    Parameters
    ----------
    levelwise_variables : list of dicts
        List of dicts containing all the variables extracted from the log
        file of all the repetitions of a subject in a specific level, on a
        specific setup.

    Returns
    -------
    list
        List of the total amount of health lost during each repetition.

    """
    total_health_lost = []
    for repetition_dict in levelwise_variables:
        health_change = np.diff(repetition_dict["health"], n=1)
        total_health_lost.append(sum([x for x in health_change if x < 0]))
    return total_health_lost


def compute_completed(levelwise_variables):
    """
    Check if the player completed the repetition or not.
    Here we use "ended the repetition without losing a life" as indicative of a
    completed level.

    Parameters
    ----------
    levelwise_variables : list of dicts
        List of dicts containing all the variables extracted from the log
        file of all the repetitions of a subject in a specific level, on a
        specific setup.

    Returns
    -------
    list
        List of 0 and 1, indicative of a repetition completed (1) or not (0).

    """
    completed = []
    for repetition_dict in levelwise_variables:
        repetition_lives = repetition_dict["lives"]
        lives_lost = sum([x for x in np.diff(repetition_lives, n=1) if x < 0])
        if lives_lost == 0:
            completed.append(1)
        else:
            completed.append(0)
    return completed


def compute_completed_perc(levelwise_variables):
    """Computes the percentage of the level that was actually completed during
    each repetition. A repetition in which the player reached a position located
    at 70% of the maximum position in a level will have a value of 70. We use
    "reach somewhere around the end of the level" as a proxy for complete,
    because boss fights introduce some jitter around the farthest position one
    can reach.

    Parameters
    ----------
    levelwise_variables : list of dicts
        List of dicts containing all the variables extracted from the log
        file of all the repetitions of a subject in a specific level, on a
        specific setup.

    Returns
    -------
    list
        List of the completion percentages for each repetition.

    """

    X_player_list = fix_position_resets(levelwise_variables)
    # Find max value of X_player across all repetitions (that means that we
    # assume that the level was completed at least once in our data)
    max_X_list = []
    for X_player in X_player_list:
        max_X_list.append(np.max(X_player))

    # We assume that the level is completed at 100% when the player reaches
    # end_of_level - 100, because there is some variance due to boss fights
    end_of_level = np.max(max_X_list) - 100

    # Now store the proportion of the level completed
    # by comparing the end position against the max position and expressing it
    # as a percent.
    completed_perc = []
    for curr_X in max_X_list:
        if curr_X > end_of_level:
            curr_X = end_of_level
        completed_perc.append(curr_X / end_of_level * 100)
    return completed_perc


def compute_time2complete(levelwise_variables):
    """
    Number of frames elapsed until the last position in the level, only for repetitions that are completed.
    Failed repetitions are replaced by the average value of completed repetitions.

    Inputs :
    levelwise_variables = dict with one entry per variable. Each entry contains List of lists, with
    level1 lists = reps and level2 lists = frames

    Outputs :
    time2complete = list with one element per repetition (number of frames until end)

    """
    time2pos_lists = get_time2pos(fix_position_resets(levelwise_variables))
    completed = compute_completed(levelwise_variables)
    time2complete = []
    for i, r in enumerate(completed):
        if r:
            time2complete.append(time2pos_lists[i][-1])
        else:
            time2complete.append(np.nan)
    time2complete = [
        np.nanmean(time2complete) if np.isnan(x) else x for x in time2complete
    ]
    return time2complete


def compute_rel_speed(levelwise_variables):
    """
    Compute the average (per repetition) relative speed based on the distribution of time2pos, i.e. the number
    of frames elapsed until a position is reached.

    Inputs :
    levelwise_variables = dict with one entry per variable. Each entry contains List of lists, with
    level1 lists = reps and level2 lists = frames

    Outputs :
    rel_speed = list with one element per repetition
    """
    X_player_list = fix_position_resets(levelwise_variables)
    time2pos_list = get_time2pos(X_player_list)
    distrib_t2p = distributions_t2p(time2pos_list)
    rel_speed = []
    for i, run in enumerate(time2pos_list):
        rel_speed.append(np.mean(compare_to_distrib(distrib_t2p, run)))
    return rel_speed


# relative speed prerequisites
def fix_position_resets(levelwise_variables):
    """Sometimes X_player resets to 0 but the player's position should keep
    increasing.
    This fixes it and makes sure that X_player is continuous. If not, the
    values after the jump are corrected.

    Parameters
    ----------
    X_player : list
        List of raw positions at each timeframe from one repetition.

    Returns
    -------
    list
        List of lists of fixed (continuous) positions. One per repetition.

    """
    X_player_list = []
    for repetition_dict in levelwise_variables:
        fixed_X_player = []
        raw_X_player = repetition_dict["X_player"]
        fix = 0  # keeps trace of the shift
        fixed_X_player.append(raw_X_player[0])  # add first frame
        for i in range(1, len(raw_X_player) - 1):  # ignore first and last frames
            if raw_X_player[i - 1] - raw_X_player[i] > 100:
                fix += raw_X_player[i - 1] - raw_X_player[i]
            fixed_X_player.append(raw_X_player[i] + fix)
        X_player_list.append(fixed_X_player)
    return X_player_list


def get_time2pos(X_player_list):
    """Count the number of frames elapsed until each (horizontal) position
    is reached for the first time.

    Parameters
    ----------
    positions : list
        Player's X position at each frame.

    Returns
    -------
    list
        Number of frames to reach each position.

    """
    time2pos_list = []
    for positions in X_player_list:
        frame_idx = 0
        time2pos = []
        for p in range(min(positions), max(positions) + 1):
            while positions[frame_idx] < p:
                frame_idx += 1
            time2pos.append(frame_idx + 1)  # +1 in case frame indices start at 1

        assert np.all(np.diff(time2pos) >= 0), "Output not monotonically increasing."
        time2pos_list.append(time2pos)

    return time2pos_list


def distributions_t2p(time2pos_list):
    """Create distribution of time2pos values at each position, across all
    repetitions of a single subject at a specific level.


    Parameters
    ----------
    time2pos_list : list
        Number of frames to reach each position, for each repetition.

    Returns
    -------
    list
        A distribution of the time2pos values for each position..

    """
    distrib_t2p = []
    for i in range(max([len(time2pos) for time2pos in time2pos_list])):
        pos_distrib = []
        for run in time2pos_list:
            try:
                pos_distrib.append(run[i])
            except:
                pass
        distrib_t2p.append(np.array(pos_distrib))
    return distrib_t2p


def distributions_p2t(X_player_list):
    """Obtains the distribution, at each frame, of the positions reached by the
    player at that frame across repetitions.

    Parameters
    ----------
    X_player_list : list
        List of (fixed) X_player position for every frame in each repetition.

    Returns
    -------
    list of list
        Distribution of player positions at a specific frame, across repetitions.

    """
    distrib_p2t = []
    for i in range(max([len(x) for x in X_player_list])): # Find the longest rep
        distrib_p2t_atframe = []
        for X_player in X_player_list:
            if i < len(X_player):
                distrib_p2t_atframe.append(X_player[i])
        distrib_p2t.append(distrib_p2t_atframe)
    return distrib_p2t


def get_ranked_p2t(distrib_p2t, X_player_repetition):
    """Short summary.

    Parameters
    ----------
    distrib_p2t : list of lists
        Distribution of player positions at a specific frame, across repetitions.
    X_player_repetition : list
        List of positions at each frame, during one repetition.

    Returns
    -------
    list
        Rank of the current position at each frame against positions of all
        other repetitions.

    """
    ranked_p2t = []
    for idx, pos in enumerate(X_player_repetition):
        ranked_p2t.append(percentileofscore(distrib_p2t[idx], pos))
    return ranked_p2t


def compare_to_distrib(distrib_t2p, time2pos_repetition):
    """Compare an individual run to the distribution of all runs and get
    Ranks time2pos values of a single repetition against the distribution of
    values across all repetitions.
    percentile for each pos. i.e. compute relative time_to_position


    Parameters
    ----------
    distrib_t2p : type
        Description of parameter `distrib_t2p`.
    time2pos_repetition : type
        Description of parameter `time2pos_repetition`.

    Returns
    -------
    type
        Description of returned object.

    """
    pos_percentile = []
    for i, pos in enumerate(time2pos_repetition):
        pos_percentile.append(percentileofscore(distrib_t2p[i], pos))
    return pos_percentile


def compute_framewise_t2p_rank(time2pos_list, distrib_t2p, X_player_list):
    """Short summary.

    Parameters
    ----------
    time2pos_list : type
        Description of parameter `time2pos_list`.
    distrib_t2p : type
        Description of parameter `distrib_t2p`.
    X_player_list : type
        Description of parameter `X_player_list`.

    Returns
    -------
    type
        Description of returned object.

    """
    framewise_t2p_list = []
    for idx, time2pos in enumerate(time2pos_list):
        pos_percentile = compare_to_distrib(distrib_t2p, time2pos)
        framewise_t2p = []
        max_pos_reached = 0
        for pos in X_player_list[idx]:
            newpos = pos - min(X_player_list[idx])
            framewise_t2p.append(pos_percentile[newpos])
        framewise_t2p_list.append(framewise_t2p)
    return framewise_t2p_list


# aps computation
def compute_framewise_aps(levelwise_variables, actions, FS=60):
    """Generates an array containing .

    Parameters
    ----------
    levelwise_variables : type
        Description of parameter `levelwise_variables`.
    actions : type
        Description of parameter `actions`.
    FS : type
        Description of parameter `FS`.

    Returns
    -------
    type
        Description of returned object.

    """
    # generate events for each of them
    action_mat = []
    for act in actions:
        var = levelwise_variables[act]
        var_bin = [int(val) for val in var]
        diffs = list(np.diff(var_bin, n=1))
        absdiffs = [abs(x) for x in diffs]
        action_mat.append(absdiffs)
    action_mat = np.array(action_mat)

    # compute number of action at each frame
    act_per_frame = np.zeros(action_mat.shape[1])
    for i in range(len(act_per_frame)):
        act_per_frame[i] = sum(action_mat[:, i])

    padded_apf = np.concatenate(
        (np.zeros(int(FS / 2)), act_per_frame, np.zeros(int(FS / 2)))
    )
    framewise_aps = np.zeros(len(padded_apf) - FS)
    for frame_idx in range(len(framewise_aps)):
        framewise_aps[frame_idx] = sum(padded_apf[frame_idx : frame_idx + FS])

    return framewise_aps


# if __name__ == "__main__":
