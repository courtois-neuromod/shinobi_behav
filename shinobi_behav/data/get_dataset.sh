# Run these lines to install both shinobi and shinobi_beh datasets
cd ./data
export AWS_ACCESS_KEY_ID=yharel  AWS_SECRET_ACCESS_KEY=qrVwnh1NYZfpCHtTL7glEfdY
datalad install git@github.com:courtois-neuromod/shinobi
cd shinobi
git checkout event_files
datalad get ./
scp -r yharel@elm.criugm.qc.ca:/data/neuromod/DATA/games/shinobi_beh ./

# Additional lines to setup gym environment
cd stimuli
datalad get ./
conda activate hyruuk_shinobi_behav
python3 -m retro.import ./data/shinobi/stimuli/ShinobiIIIReturnOfTheNinjaMaster-Genesis
