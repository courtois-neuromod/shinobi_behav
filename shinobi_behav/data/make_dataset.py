import logging
import shinobi_behav
from shinobi_behav.data.data import get_levelreps
import os.path as op
import pickle
import os
from shinobi_behav.parsing import parser
from shinobi_behav.utils import pickle_save


def main():
    """ Extracts variables from bk2 files and stores them as lists of
    dictionnaries, splitted by subject, setup and level, and sorted by date.
    This step is a prerequisite for features computation.
    """
    #args = parser.parse_args()
    path_to_data = shinobi_behav.DATA_PATH
    subjects = shinobi_behav.SUBJECTS
    levels = [f"level-{lev}" for lev in shinobi_behav.LEVELS]
    logger = logging.getLogger(__name__)
    logger.info("Processing datasets for at-home VS in-scanner analysis.")
    if not op.isdir(op.join(path_to_data, "processed")):
        os.mkdir(op.join(path_to_data, "processed"))
        logger.info("Directory created")
    else:
        logger.info("Directory already exists")
    # Start loop
    for subj in subjects:
        for level in levels:
            for setup in ["scan", "home"]:
                level_variables_path = op.join(
                    path_to_data, "processed", f"{subj}_{level}_levelwise_variables_{setup}.pkl"
                )
                if not os.path.exists(level_variables_path):
                    logger.info(f"Extracting game variables for {subj}_level-{level}")
                    logger.info("Training sessions ({})".format(setup))
                    # Extract variables and store them by sub*level*setup
                    level_variables = get_levelreps(
                        path_to_data, subj, level, remove_fake_reps=True, setup=setup
                    )

                    pickle_save(level_variables_path, level_variables)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
