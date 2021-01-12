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


 # Set constants
sub = 'sub-01'
actions = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
dpath = path_to_data + 'shinobi/'



seslist= os.listdir(dpath + sub)

# load nifti imgs
fmri_imgs = []
confounds = []
for ses in sorted(seslist):
    runs = [filename[-13] for filename in os.listdir(dpath + '{}/{}/func'.format(sub, ses)) if 'bold.nii.gz' in filename]
    print('Processing {}'.format(ses))
    print(runs)
    for run in sorted(runs):
        print('run : {}'.format(run))
        data_fname = dpath + 'derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, sub, ses, run)
        confounds_fname = dpath + 'derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_desc-confounds_timeseries.tsv'.format(sub, ses, sub, ses, run)
        anat_fname = '/project/rrg-pbellec/hyruuk/hyruuk_shinobi_behav/data/anat/derivatives/fmriprep-20.2lts/fmriprep/{}/anat/{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(sub, sub)
        fmri_img = image.concat_imgs(data_fname)
        masker = NiftiMasker()
        masker.fit(anat_fname)
        confounds.append(pd.DataFrame.from_records(load_confounds.Params36().load(confounds_fname)))
        fmri_img_conf = masker.transform(fmri_img, confounds=confounds)

        fmri_imgs.append(fmri_img_conf)

    # load events
    with open(dpath + '{}_{}_events_files.pkl'.format(sub, ses), 'rb') as f:
        allruns_events.append(pickle.load(f))


    # build model
    print('Fitting a GLM')
    fmri_glm = FirstLevelModel(t_r=1.49,
                               noise_model='ar1',
                               standardize=False,
                               hrf_model='spm',
                               drift_model=None,
                               high_pass=.01,
                               n_jobs=16,
                               smoothing_fwhm=5,
                               mask_img=anat_fname)
    fmri_glm = fmri_glm.fit(fmri_imgs, allruns_events, confounds=confounds)
    report = fmri_glm.generate_report(contrasts=['LeftH-RightH'])
    report.save_as_html(figures_path + '/{}_{}_LmR_flm.html'.format(sub, ses))

    # get stats map
    z_map = fmri_glm.compute_contrast(['LeftH-RightH'],
        output_type='z_score', stat_type='F')

    # compute thresholds
    clean_map, threshold = map_threshold(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)
    uncorr_map, threshold = map_threshold(z_map, alpha=.001, height_control='fpr')

    # save images
    print('Generating views')
    view = plotting.view_img(clean_map, threshold=3, title='Left minus Right Hand (FDR<0.05), Noyaux > 10 voxels')
    view.save_as_html(figures_path + '/{}_{}_LmR_statsmap_allruns_FDRcluster_fwhm5.html'.format(sub, ses))
    # save also uncorrected map
    view = plotting.view_img(uncorr_map, threshold=3, title='Left minus Right Hand (p<0.001), uncorr')
    view.save_as_html(figures_path + '/{}_{}_LmR_statsmap_allruns_uncorr_fwhm5.html'.format(sub, ses))
