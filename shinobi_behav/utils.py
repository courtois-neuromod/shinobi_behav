import pickle


def pickle_save(fpath, object_to_save):
    """
    Saves the python object in a pickle file

    Parameters
    ----------
    fpath : str
        Path to the file to be created
    content : str
        Python object
    """
    with open(fpath, "wb") as f:
        pickle.dump(object_to_save, f)


def list_save(fpath, list_to_save):
    """Saves the python list in a text file, with items separated by lines.

    Parameters
    ----------
    fpath : str
        Path to the file to be created
    list : list
        Python list
    """
    with open(fpath, "w") as f:
        for element in list_to_save:
            f.write(str(element) + "\n")


def filter_run(repetition_variable, order=3, cutoff=0.005):
    """Filter a vector in the time dimension. Can be used on the relative
    time_to_position for visualization and computation of the derivative.

    Parameters
    ----------
    repetition_variable : List
        List (vector) of values of a variable at each frame of a repetition.
    order : int
        Filter order.
    cutoff : float
        Filter cutoff.

    Returns
    -------
    list
        Filtered vector.

    """
    b, a = signal.butter(order, cutoff)
    variable_filtered = signal.filtfilt(b, a, repetition_variable)
    return variable_filtered


def moving_descriptive(perf_measure, win_size=3, metric="mean"):
    """Run a moving window across repetitions and compute a descriptive metrics.

    Parameters
    ----------
    perf_measure : list
        List of the performance measure values for each repetition.
    win_size : int
        Size (in number of samples) of the moving window.
    metric : str
        Metric to apply. Can be "mean", "median" or "std".

    Returns
    -------
    list
        List of values averaged across the moving windows.

    """
    perf_measure = np.array(perf_measure)
    idx = np.arange(win_size) + np.arange(len(perf_measure) - win_size + 1)[:, None]
    if metric == "mean":
        out = list(np.mean(perf_measure[idx], axis=1))
    if metric == "median":
        out = list(np.median(perf_measure[idx], axis=1))
    if metric == "std":
        out = list(np.std(perf_measure[idx], axis=1))
    return out
