import logging
import shinobi_behav
from shinobi_behav.features.features import load_features_dict
import click
from shinobi_behav.visualization.visualization import lcurves_plot
import matplotlib.pyplot as plt
import os.path as op
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='04',
    type=str,
    help="Subject to process",
)
parser.add_argument(
	"-l",
	"--level",
	default = '1-0',
	type = str,
	help="Level to process"
)

args = parser.parse_args()


def main():
    """ Create repetition-wise features based on the processed data from (../processed)
    These features are then used to plot the learning curves with src/visualization/generate_lcurves.py.
    src.features.build_features.py must be ran before using generate_lcurves
    """
    path_to_data = shinobi_behav.DATA_PATH
    logger = logging.getLogger(__name__)
    logger.info('create repetition-wise features')
    variables = ['Health loss', 'Max score', 'Percent complete']

    subj = args.subject
    level = args.level

    print('Extracting game variables for {}_level-{}'.format(subj, level))

    data_dict = load_features_dict(path_to_data, subj, level, 'home', save=True, metric='mean')

    # Generate and save plot
    fig = lcurves_plot(data_dict, variables, '{}_level{}'.format(subj, level))
    fig.savefig(op.join(figures_path, f'{subj}_{level}_learning_curve.tif', dpi=300, bbox_inches='tight'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
