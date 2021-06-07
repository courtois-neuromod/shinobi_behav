# -*- coding: utf-8 -*-
import logging
from src.params import *
from data import combine_variables


def main():
    """ Runs data processing scripts to turn raw data from (../bids) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # start loop
    for subj in subjects:
        for level in levels:
            print('Extracting game variables for {}_level-{}'.format(subj, level))
            allvars = combine_variables(path_to_data, subj, level, behav=False, save=False)
            allvars = remove_fake_reps(allvars)
            allvars_path = op.join(path_to_data, 'processed','{}_{}_allvars_scan.pkl'.format(subject, level))
            with open(allvars_path, 'wb') as f:
                pickle.dump(allvars, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
