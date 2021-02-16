from src.params import actions, path_to_data
from src.data.data import retrieve_scanvariables
from src.features.annotations import create_runevents
import pickle
import os
dpath = '/media/hyruuk/Seagate Expansion Drive/DATA/shinobi/'
import logging
import pandas as pd


subjects = ['sub-01']
sessions = os.listdir(dpath + 'sub-01')



def main():
    for sub in subjects:
        for ses in sorted(sessions):
            runs = [filename[-13] for filename in os.listdir(dpath + '{}/{}/func'.format(sub, ses)) if 'bold.nii.gz' in filename]
            for run in sorted(runs):
                events_fname = dpath + '{}/{}/func/{}_{}_task-shinobi_run-0{}_events.tsv'.format(sub, ses, sub, ses, run)
                startevents = pd.read_table(events_fname)
                files = startevents['stim_file'].values.tolist()
                files = [dpath + file for file in files]
                # Retrieve variables from these files
                runvars = retrieve_scanvariables(files)
                events_df = create_runevents(runvars, startevents, actions=actions)
                events_path = path_to_data + 'processed/annotations/{}_{}_run-0{}.pkl'.format(sub, ses, run)
                with open(events_path, 'wb') as f:
                    pickle.dump(events_df, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
