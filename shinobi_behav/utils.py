import pickle


def pickle_save(fpath, content):
    """
    Saves the content in a pickle file

    Parameters
    ----------
    fpath : str
        Path to the file to be created
    content : str
        Python object
    """
    with open(fpath, "wb") as f:
        pickle.dump(content, f)
