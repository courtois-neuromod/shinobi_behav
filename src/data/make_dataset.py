# -*- coding: utf-8 -*-
import logging
from src.params import *
from data import combine_variables


def main():
    """ Runs data processing scripts to turn raw data from (../bids) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Processing datasets for learning visualization.')

    # start loop
    for subj in subjects:
        for level in levels:
            print('Extracting game variables for {}_level-{}'.format(subj, level))
            print('Training sessions (NUC)')
            allvars_behav = combine_variables(path_to_data, subj, level, behav=True, save=False)
            allvars_behav = remove_fake_reps(allvars_behav)
            allvars_behav_path = op.join(path_to_data, 'processed','{}_{}_allvars_behav.pkl'.format(subject, level))
            with open(allvars_scan_path, 'wb') as f:
                pickle.dump(allvars_scan, f)
            print('Scan sessions')
            allvars_scan = combine_variables(path_to_data, subj, level, behav=False, save=False)
            allvars_scan = remove_fake_reps(allvars_behav)
            allvars_scan_path = op.join(path_to_data, 'processed','{}_{}_allvars_scan.pkl'.format(subject, level))
            with open(allvars_scan_path, 'wb') as f:
                pickle.dump(allvars_scan, f)
            print('Done.')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
