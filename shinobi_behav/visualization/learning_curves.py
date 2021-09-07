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
    path_to_data = shinobi_behav.path_to_data
    logger = logging.getLogger(__name__)
    logger.info('create repetition-wise features')
    usable_filenames_all = {}
    # start loop
    for subj in subjects:
        plotlabels = True
        for level in levels:
            print('Extracting game variables for {}_level-{}'.format(subj, level))

            data_dict = load_features_dict(path_to_data, subj, level, save=True, metric='mean')

            # Generate individual plots and obtain threshold
            variables = ['Health loss', 'Max score', 'Percent complete']
            fig, axes = plt.subplots(len(variables), 1, figsize=(5,20))
            idx_thresh_all = []

            for idx, var in enumerate(variables):
                ax, idx_thresh = learning_curve(data_dict, 'Days of training', var,
                                            zscore=False, plot=True,
                                            x_jitter=1, y_jitter=0.1,
                                            ax=axes[idx], plotlabels=plotlabels, threshold = None, curves=[])
                idx_thresh_all.append(idx_thresh)
            plotlabels = False
            median_thresh = idx_thresh_all[2]

            # Adjust overall plots
            for ax in fig.get_axes():
                ax.label_outer()
            fig.suptitle('{}_level{}'.format(subj, level), y=0.92, fontsize=20)
            fig.savefig(figures_path + '/{}_{}_learning_curve.tif'.format(subj, level), dpi=300, bbox_inches='tight')

            # Get list of "useable repetitions" for models training
            data_dict = load_features_dict(path_to_data, subj, level, save=True, metric=None)
            allvars = combine_variables(path_to_data, subj, level)
            allvars = remove_fake_reps(allvars)
            usable_filenames = fetch_usable_reps(allvars, data_dict, median_thresh)
            usable_filenames_all['{}_{}'.format(subj, level)] = usable_filenames
    usable_filenames_path = op.join(path_to_data, 'postthreshold_filenames_bymedian.mat')

    with open(usable_filenames_path, 'wb') as f:
        pickle.dump(usable_filenames_all, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
