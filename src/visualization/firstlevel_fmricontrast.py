import pandas as pd
import os.path as op
import retro
from src.features.annotations import generate_key_events, generate_aps_events, plot_bidsevents
from src.features.features import compute_framewise_aps
import matplotlib.pyplot as plt
from src.params import figures_path
from nilearn import plotting
from nilearn import image
import os
import numpy as np
from nilearn.plotting import plot_design_matrix


def retrieve_variables(files):
    '''
    files : list of files with complete path

    variable_lists : dictionnary (each variable is an entry) containing list of arrays of
    length corresponding to the number of frames in each run,
    with runs ordered by timestamp.
    '''

    variables_lists = {}

    for file in files:
        level = file[-11:-8]
        timestamp = file[-73:-65]
        print(file)
        if level == '5-0':
            env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level5')
        else:
            env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level'+level)
        actions = env.buttons

        run_variables = {}
        key_log = retro.Movie(file)
        env.reset()
        run_completed = False
        while key_log.step():
            a = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
            _,_,done,i = env.step(a)

            if variables_lists == {}: # init final dict
                variables_lists['filename'] = []
                variables_lists['timestamp'] = []
                variables_lists['level'] = []
                for action in actions:
                    variables_lists[action] = []
                for variable in i.keys():
                    variables_lists[variable] = []

            if run_variables == {}: # init temp dict
                for variable in i.keys():
                    run_variables[variable] = []
                for action in actions:
                    run_variables[action] = []

            for variable in i.keys(): # fill up temp dict
                run_variables[variable].append(i[variable])
            for idx_a, action in enumerate(actions):
                run_variables[action].append(a[idx_a])

            if done == True:
                run_completed = True
        variables_lists['filename'].append(file)
        variables_lists['timestamp'].append(timestamp)
        variables_lists['level'].append(level)

        for variable in run_variables.keys():
            variables_lists[variable].append(run_variables[variable])
        env.close()
    return variables_lists

def create_runevents(runvars, startevents, actions, FS=60, min_dur=1, get_aps=True, get_actions=True):
    onset_reps = startevents['onset'].values.tolist()
    dur_reps = startevents['duration'].values.tolist()
    lvl_reps = [x[-11] for x in startevents['stim_file'].values.tolist()]

    if get_aps:
        framewise_aps = compute_framewise_aps(runvars, actions=actions, FS=FS)

    # init df list
    all_df = []

    for idx, onset_rep in enumerate(onset_reps):
        print('Extracting events for {}'.format(runvars['filename'][idx]))
        if get_actions:
            # get the different possible actions
            # generate events for each of them
            for act in actions:
                var = runvars[act][idx]
                temp_df = generate_key_events(var, act, FS=FS)
                temp_df['onset'] = temp_df['onset'] + onset_rep
                temp_df['trial_type'] = lvl_reps[idx] + '_' + temp_df['trial_type']
                all_df.append(temp_df)
        if get_aps:
            temp_df = generate_aps_events(framewise_aps[idx], FS=FS)
            temp_df['onset'] = temp_df['onset'] + onset_rep
            temp_df['trial_type'] = lvl_reps[idx] + '_' + temp_df['trial_type']
            all_df.append(temp_df)


    events_df = pd.concat(all_df).sort_values(by='onset').reset_index(drop=True)
    return events_df




 # Set constants
sub = 'sub-01'
actions = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
dpath = '/media/hyruuk/Seagate Expansion Drive/DATA/data/shinobi/'



seslist= os.listdir(dpath + sub)

allruns_events = []
fmri_img = []
for ses in seslist:
    runs = [filename[-13] for filename in os.listdir(dpath + '{}/{}/func'.format(sub, ses)) if 'bold.nii.gz' in filename]
    print(runs)
    for run in runs:
        print('computing run {}'.format(run))
        events_fname = dpath + '{}/{}/func/{}_{}_task-shinobi_run-0{}_events.tsv'.format(sub, ses, sub, ses, run)
        filename = dpath + 'derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/sub-01_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, ses, run)
        # Obtain list of bk2 files from events
        startevents = pd.read_table(events_fname)
        files = startevents['stim_file'].values.tolist()
        files = [dpath + file for file in files]

        # Retrieve variables from these files
        runvars = retrieve_variables(files)
        events_df = create_runevents(runvars, startevents, actions=actions)
        events_df['trial_type'].unique()

        # Create LvR_df
        lh_df = pd.concat([events_df[events_df['trial_type'] == '1_LEFT'],
                           events_df[events_df['trial_type'] == '1_RIGHT'],
                           events_df[events_df['trial_type'] == '1_DOWN'],
                           events_df[events_df['trial_type'] == '1_UP'],
                           events_df[events_df['trial_type'] == '4_LEFT'],
                           events_df[events_df['trial_type'] == '4_RIGHT'],
                           events_df[events_df['trial_type'] == '4_DOWN'],
                           events_df[events_df['trial_type'] == '4_UP'],
                            events_df[events_df['trial_type'] == '5_LEFT'],
                           events_df[events_df['trial_type'] == '5_RIGHT'],
                           events_df[events_df['trial_type'] == '5_DOWN'],
                           events_df[events_df['trial_type'] == '5_UP']
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_df['trial_type'] = 'LeftH'

        rh_df = pd.concat([events_df[events_df['trial_type'] == '1_B'],
                           events_df[events_df['trial_type'] == '1_C'],
                           events_df[events_df['trial_type'] == '4_B'],
                           events_df[events_df['trial_type'] == '4_C'],
                            events_df[events_df['trial_type'] == '5_B'],
                           events_df[events_df['trial_type'] == '5_C']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_df['trial_type'] = 'RightH'

        LvR_df = pd.concat([lh_df, rh_df]).sort_values(by='onset').reset_index(drop=True)

        allruns_events.append(LvR_df)

        fmri_img.append(image.concat_imgs(filename))

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
