# -*- coding: utf-8 -*-
import logging
import shinobi_behav
from shinobi_behav.data.data import get_levelreps
import os.path as op
import pickle
import os


def main():
    """ Extracts variables from bk2 files and stores them as lists of dictionnaries,
    splitted by subject and by level, and sorted by date. This step is a prerequisite for
    features computation intended for the analysis of learning progression.
    """
    path_to_data = shinobi_behav.DATA_PATH
    logger = logging.getLogger(__name__)
    logger.info('Processing datasets for at-home VS in-scanner analysis.')
    if not op.isdir(op.join(path_to_data, 'processed')):
        os.mkdir(op.join(path_to_data, 'processed'))
        logger.info('Directory created')
    else:
        logger.info('Directory already exists')

    for subj in shinobi_behav.SUBJECTS:
        for level in shinobi_behav.LEVELS:
            for setup in ['scan', 'home']:
                level_variables_path = op.join(path_to_data, 'processed','{}_{}_allvars_{}.pkl'.format(subj, level, setup))
                if not os.path.exists(level_variables_path):
                    logger.info('Extracting game variables for {}_level-{}'.format(subj, level))
                    logger.info('Training sessions ({})'.format(setup))
                    level_variables = get_levelreps(path_to_data, subj, level, remove_fake_reps=True, setup=setup)
                    with open(level_variables_path, 'wb') as f:
                        pickle.dump(repetition_variables, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
