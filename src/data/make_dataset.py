# -*- coding: utf-8 -*-
import logging
from src.params import *
from data import combine_variables, remove_fake_reps
import os.path as op
import argparse

def main():
    """ Runs data processing scripts to turn raw data from (../bids) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating data dict from raw data')

    parser = argparse.ArgumentParser(description='Generates a dataset from raw .bk2 files.')
    parser.add_argument('-n', '--nuc',
                        action='store_true',
                        dest='nuc',
                        help='Use NUC files (behaviour only)'
                        )
    parser.add_argument('-s', '--scan',
                        action='store_true',
                        dest='scan',
                        help='Use files related to scan sessions.'
                        )

    args = parser.parse_args()


    # start loop
    for subj in subjects:
        for level in levels:
            if args.scan:
                print('Extracting game variables (scans) for {}_level-{}'.format(subj, level))
                allvars = combine_variables(path_to_data, subj, level, behav=False, save=False)
                allvars = remove_fake_reps(allvars)
                allvars_path = op.join(path_to_data, 'processed','{}_{}_allvars_scan.pkl'.format(subj, level))
            elif args.nuc:
                print('Extracting game variables (NUC) for {}_level-{}'.format(subj, level))
                allvars = combine_variables(path_to_data, subj, level, behav=True, save=False)
                allvars = remove_fake_reps(allvars)
                allvars_path = op.join(path_to_data, 'processed','{}_{}_allvars_nuc.pkl'.format(subj, level))
            else:
                print('Aborted.')
                print('Please chose between --nuc and --scan arguments.')
                quit()
            if not op.isdir(op.join(path_to_data, 'processed')):
                os.mkdir(op.join(path_to_data, 'processed'))
            with open(allvars_path, 'wb') as f:
                pickle.dump(allvars, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
