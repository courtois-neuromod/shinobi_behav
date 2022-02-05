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
    """Short summary.

    Parameters
    ----------
    fpath : str
        Path to the file to be created
    list : list
        Python list
    """
    with open(fpath, "wb") as f:
        for element in list_to_save:
            f.write(element + "\n")
