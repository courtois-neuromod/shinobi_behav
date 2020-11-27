import pandas as pd
import os.path as op

from src.features.annotations import generate_key_events, generate_aps_events, plot_bidsevents
from src.features.features import compute_framewise_aps
import matplotlib.pyplot as plt
from src.params import figures_path
from nilearn import plotting
from nilearn import image
import os
import numpy as np
from nilearn.plotting import plot_design_matrix
from nistats.thresholding import map_threshold
import pickle






 # Set constants
sub = 'sub-01'
actions = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
dpath = '/project/rrg-pbellec/hyruuk/hyruuk_shinobi_behav/data/shinobi/'



seslist= os.listdir(dpath + sub)

allruns_events = []
fmri_img = []
for ses in seslist:
    runs = [filename[-13] for filename in os.listdir(dpath + '{}/{}/func'.format(sub, ses)) if 'bold.nii.gz' in filename]
    print(runs)
    for run in runs:
        filename = dpath + 'derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/sub-01_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, ses, run)
        fmri_img.append(image.concat_imgs(filename))

# load events
with open(dpath + '{}_events_files.pkl', 'rb') as f:
    allruns_events = pickle.load(f)


# build model
from nilearn.glm.first_level import FirstLevelModel
print('Fitting a GLM')
fmri_glm = FirstLevelModel(t_r=1.49,
                           noise_model='ar1',
                           standardize=False,
                           hrf_model='spm',
                           drift_model='cosine',
                           high_pass=.01)
fmri_glm = fmri_glm.fit(fmri_img, allruns_events)
mean_img = image.mean_img(fmri_img)

# get stats map
z_map = fmri_glm.compute_contrast(['LeftH-RightH'] * len(fmri_img),
    output_type='z_score', stat_type='F')

# compute thresholds
clean_map, threshold = map_threshold(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)
uncorr_map, threshold = map_threshold(z_map, alpha=.001, height_control='fpr')

# save images
view = plotting.view_img(clean_map, threshold=3, title='Left minus Right Hand (FDR=0.05), Noyaux > 10 voxels')
view.save_as_html(figures_path + '/{}_LmR_statsmap_allruns_FDRcluster.html'.format(sub))

view = plotting.view_img(clean_map, threshold=3, title='Left minus Right Hand (p<0.001), uncorr')
view.save_as_html(figures_path + '/{}_LmR_statsmap_allruns_uncorr.html'.format(sub))
