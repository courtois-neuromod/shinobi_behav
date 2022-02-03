#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_make_dataset
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

module load python/3.8.0

/home/hyruuk/python_envs/shinobi_env/bin/python /project/def-pbellec/hyruuk/shinobi_fmri/data/shinobi_behav/shinobi_behav/data/make_dataset.py -s $1 -l $2
