import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from src.params import figures_path, path_to_data
from nilearn import plotting
from nilearn import image
import os
import numpy as np
from nilearn.plotting import plot_design_matrix
from nistats.thresholding import map_threshold
from nilearn.glm.first_level import FirstLevelModel
from nilearn.input_data import NiftiMasker
import load_confounds
import pickle
import nilearn
from scipy import signal
from scipy.stats import zscore





 # Set constants
sub = 'sub-01'
actions = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
dpath = path_to_data + 'shinobi/'



seslist= os.listdir(dpath + sub)
cmaps = []
# load nifti imgs
for ses in sorted(seslist):
    cmap_name = dpath + 'processed/cmaps/LeftH-RightH/{}_{}.nii.gz'.format(sub, ses)
    cmaps.append(cmap_name)


second_level_input = cmaps
second_design_matrix = pd.DataFrame([1] * len(second_level_input),
                             columns=['intercept'])

from nilearn.glm.second_level import SecondLevelModel
second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
second_level_model = second_level_model.fit(second_level_input,
                                            design_matrix=second_design_matrix)

z_map = second_level_model.compute_contrast(output_type='z_score')

report = second_level_model.generate_report(contrasts=['intercept'])
report.save_as_html(figures_path + '/{}_LmR_slm.html'.format(sub))

# compute thresholds
clean_map, threshold = map_threshold(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)
uncorr_map, threshold = map_threshold(z_map, alpha=.001, height_control='fpr')

# save images
print('Generating views')
view = plotting.view_img(clean_map, threshold=3, title='Left minus Right Hand (FDR<0.05), Noyaux > 10 voxels')
view.save_as_html(figures_path + '/{}_LmR_slm_FDRcluster_fwhm5.html'.format(sub))
# save also uncorrected map
view = plotting.view_img(uncorr_map, threshold=3, title='Left minus Right Hand (p<0.001), uncorr')
view.save_as_html(figures_path + '/{}_{}_LmR_slm_uncorr_fwhm5.html'.format(sub))
