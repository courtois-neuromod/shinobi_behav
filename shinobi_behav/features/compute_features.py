# -*- coding: utf-8 -*-
import logging
import shinobi_behav
from shinobi_behav.data.data import get_levelreps
from features import load_features_dict
import click


def main():
    """ Create averaged features based on the datadicts in ./data/processed/
    These features are then used to plot the learning curves with src/visualization/generate_lcurves.py
    """
    path_to_data = shinobi_behav.DATA_PATH
    logger = logging.getLogger(__name__)
    logger.info('Create repetition-wise features')

    setup='home'
    # start loop
    for subj in shinobi_behav.SUBJECTS:
        for level in shinobi_behav.LEVELS:
            logger.info('Extracting game variables for {}_level-{}'.format(subj, level))
            allvars = get_levelreps(path_to_data, subj, level, remove_fake_reps=True, setup=setup)
            load_features_dict(path_to_data, subj, level, save=True, metric='mean', allvars=allvars)
            load_features_dict(path_to_data, subj, level, save=True, metric=None, allvars=allvars)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
