# -*- coding: utf-8 -*-
import logging
from src.params import *
from src.data.data import combine_variables, remove_fake_reps
from features import load_features_dict
import click


def main():
    """ Create repetition-wise features based on the processed data from (../processed)
    These features are then used to plot the learning curves with src/visualization/generate_lcurves.py
    """
    logger = logging.getLogger(__name__)
    logger.info('create repetition-wise features')

    # start loop
    for subj in subjects:
        for level in levels:
            print('Extracting game variables for {}_level-{}'.format(subj, level))
            allvars = combine_variables(path_to_data, subj, level)
            allvars = remove_fake_reps(allvars)
            load_features_dict(path_to_data, subj, level, save=True, metric='mean', allvars=allvars)
            load_features_dict(path_to_data, subj, level, save=True, metric=None, allvars=allvars)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
