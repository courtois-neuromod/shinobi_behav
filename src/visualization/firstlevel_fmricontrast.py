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
# load nifti imgs
for ses in sorted(seslist):
    runs = [filename[-13] for filename in os.listdir(dpath + '{}/{}/func'.format(sub, ses)) if 'bold.nii.gz' in filename]
    fmri_imgs = []
    design_matrices = []
    confounds = []
    confounds_cnames = []
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
        fmri_imgs.append(fmri_img)
        conf=load_confounds.Params36()
        conf.load(confounds_fname)
        confounds_cnames.append(conf.columns_)
    # load events
    with open(path_to_data + '{}_{}_events_files.pkl'.format(sub, ses), 'rb') as f:
        allruns_events = pickle.load(f)


    # find the shortest run and retain it's number of confounds
    conf_lengths = []
    for con in confounds:
        conf_lengths.append(con.shape[1])
    conf_minlen = np.min(conf_lengths)
    # create design matrices
    for idx, run in enumerate(sorted(runs)):
        t_r = 1.49
        n_slices = confounds[idx].shape[0]
        frame_times = np.arange(n_slices) * t_r

        new_idx_con = 0
        new_confounds = []
        new_confounds_cnames = []
        for idx_con, con in enumerate(np.asarray(confounds[idx]).squeeze().T):
            if idx_con < conf_minlen-12 or idx_con >= confounds[idx].shape[1]-12:
                new_confounds.append(con)
                new_confounds_cnames.append(confounds_cnames[idx][idx_con])
                new_idx_con = new_idx_con + 1

        print(confounds[idx].shape)
        print(np.array(new_confounds).shape)
        print(len(new_confounds_cnames))
        print(type(confounds[idx]))

        design_matrix = nilearn.glm.first_level.make_first_level_design_matrix(frame_times,
                                                                               events=allruns_events[idx],
                                                                              drift_model=None,
                                                                              add_regs=new_confounds,
                                                                              add_reg_names=new_confounds_cnames)
        LeftH_ts = np.asarray(design_matrix['LeftH'])
        RightH_ts = np.asarray(design_matrix['RightH'])

        b, a = signal.butter(3, 0.01, btype='high')
        LeftH_ts_hpf = signal.filtfilt(b, a, LeftH_ts)
        RightH_ts_hpf = signal.filtfilt(b, a, RightH_ts)
        LeftH_ts_hpf_z = zscore(LeftH_ts_hpf)
        RightH_ts_hpf_z = zscore(RightH_ts_hpf)

        design_matrix['LeftH'] = LeftH_ts_hpf_z
        design_matrix['RightH'] = RightH_ts_hpf_z

        design_matrices.append(design_matrix)



    # build model
    try:
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
        fmri_glm = fmri_glm.fit(fmri_imgs, design_matrices=design_matrices)
        report = fmri_glm.generate_report(contrasts=['LeftH-RightH'])
        report.save_as_html(figures_path + '/{}_{}_LmR_flm-removedconfs.html'.format(sub, ses))

        # get stats map
        z_map = fmri_glm.compute_contrast(['LeftH-RightH'],
            output_type='z_score', stat_type='F')

        # compute thresholds
        clean_map, threshold = map_threshold(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)
        uncorr_map, threshold = map_threshold(z_map, alpha=.001, height_control='fpr')

        # save images
        print('Generating views')
        view = plotting.view_img(clean_map, threshold=3, title='Left minus Right Hand (FDR<0.05), Noyaux > 10 voxels')
        view.save_as_html(figures_path + '/{}_{}_LmR_flm-removedconfs_allruns_FDRcluster_fwhm5.html'.format(sub, ses))
        # save also uncorrected map
        view = plotting.view_img(uncorr_map, threshold=3, title='Left minus Right Hand (p<0.001), uncorr')
        view.save_as_html(figures_path + '/{}_{}_LmR_flm-removedconfs_allruns_uncorr_fwhm5.html'.format(sub, ses))
    except Exception as e: print('--------------MODEL NOT COMPUTED----------------' + e)
