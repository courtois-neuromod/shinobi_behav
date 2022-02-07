# Run these lines to install both shinobi and shinobi_beh datasets
cd ./data
git clone git@github.com:courtois-neuromod/shinobi_training
datalad get ./shinobi_training
git clone git@github.com:courtois-neuromod/shinobi
cd shinobi
git checkout event_files
datalad get .
cd ..
git clone git@github.com:courtois-neuromod/shinobi.stimuli
datalad get ./shinobi.stimuli

# Additional lines to setup gym environment
python3 -m retro.import ./data/shinobi.stimuli/ShinobiIIIReturnOfTheNinjaMaster-Genesis
