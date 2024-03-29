import logging
import shinobi_behav
from shinobi_behav.features.features import load_features_dict
import click
from visualization import learning_curve, fetch_usable_reps
import matplotlib.pyplot as plt
from src.data.data import combine_variables, remove_fake_reps
import os.path as op
import pickle

def main():
    """ Create repetition-wise features based on the processed data from (../processed)
    These features are then used to plot the learning curves with src/visualization/generate_lcurves.py.
    src.features.build_features.py must be ran before using generate_lcurves
    """
    path_to_data = shinobi_behav.DATA_PATH
    logger = logging.getLogger(__name__)
    logger.info('create repetition-wise features')
    variables = ['Health loss', 'Max score', 'Percent complete']

    usable_filenames_all = {}
    # start loop
    for subj in subjects:
        for level in levels:
            print('Extracting game variables for {}_level-{}'.format(subj, level))

            data_dict = load_features_dict(path_to_data, subj, level, 'home', save=True, metric='mean')

            # Generate and save plot
            fig = lcurves_plot(data_dict, variables, '{}_level{}'.format(subj, level))
            fig.savefig(op.join(figures_path, f'{subj}_{level}_learning_curve.tif', dpi=300, bbox_inches='tight'))
'''
            # Get list of "useable repetitions" for models training
            data_dict = load_features_dict(path_to_data, subj, level, save=True, metric=None)
            allvars = combine_variables(path_to_data, subj, level)
            allvars = remove_fake_reps(allvars)
            usable_filenames = fetch_usable_reps(allvars, data_dict, median_thresh)
            usable_filenames_all['{}_{}'.format(subj, level)] = usable_filenames
    usable_filenames_path = op.join(path_to_data, 'postthreshold_filenames_bymedian.mat')

    with open(usable_filenames_path, 'wb') as f:
        pickle.dump(usable_filenames_all, f)
'''
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
