from shinobi_behav.params import actions, path_to_data, subjects
from shinobi_behav.data.data import retrieve_scanvariables, extract_variables
from shinobi_behav.annotations.annotations import create_runevents
import pickle
import os
import logging
import pandas as pd


def main():
    for sub in subjects:
        sessions = os.listdir(path_to_data + 'shinobi/' + sub)
        for ses in sorted(sessions):
            runs = [filename[-13] for filename in os.listdir(path_to_data + 'shinobi/' + '{}/{}/func'.format(sub, ses)) if 'bold.nii.gz' in filename]
            for run in sorted(runs):
                events_fname = path_to_data + 'shinobi/{}/{}/func/{}_{}_task-shinobi_run-0{}_events.tsv'.format(sub, ses, sub, ses, run)
                startevents = pd.read_table(events_fname)
                files = startevents['stim_file'].values.tolist()
                # Retrieve variables from these files
                runvars = []
                for idx, file in enumerate(files):
                    if isinstance(file, str):
                        filepath = path_to_data + 'shinobi/' + file
                        #print(filepath)
                        repvars = extract_variables(filepath)
                        repvars['onset'] = startevents['onset'][idx]
                        repvars['duration'] = startevents['duration'][idx]
                        runvars.append(repvars)
                events_df = create_runevents(runvars, startevents, actions=actions)
                events_path = path_to_data + 'processed/annotations/{}_{}_run-0{}.pkl'.format(sub, ses, run)
                if not os.path.isdir(path_to_data + 'processed/annotations'):
                    os.makedirs(path_to_data + 'processed/annotations')
                with open(events_path, 'wb') as f:
                    pickle.dump(events_df, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
