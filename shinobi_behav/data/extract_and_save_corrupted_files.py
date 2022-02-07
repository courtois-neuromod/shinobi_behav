import os.path as op
import pickle
import shinobi_behav
from shinobi_behav.utils import list_save


def main():
    """Reads the aggregated datafiles from data/processed and save text lists of
    corrupted files.
    These lists will be stored in
    data/processed/{setup}_{emptyfiles or fakereps}.txt
    """
    for setup in ["home", "scan"]:
        emptyfiles = []
        fakereps = []
        for subj in shinobi_behav.SUBJECTS:
            for level in ["1", "4", "5"]:
                varfile_fpath = op.join(
                    shinobi_behav.DATA_PATH,
                    "processed",
                    f"{subj}_{level}_allvars_{setup}.pkl",
                )
                with open(varfile_fpath, "rb") as f:
                    _, names_fakereps, names_emptyfiles = pickle.load(f)
                fakereps.append(names_fakereps)
                emptyfiles.append(names_emptyfiles)
        emptyfiles_fname = op.join(
            shinobi_behav.DATA_PATH, "processed", f"{setup}_emptyfiles.txt"
        )
        fakereps_fname = op.join(
            shinobi_behav.DATA_PATH, "processed", f"{setup}_fakereps.txt"
        )
        emptyfiles = [item for flist in emptyfiles for item in flist]
        fakereps = [item for flist in fakereps for item in flist]
        list_save(emptyfiles_fname, emptyfiles)
        list_save(fakereps_fname, fakereps)


if __name__ == "__main__":
    main()
