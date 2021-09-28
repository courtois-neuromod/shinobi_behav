#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shinobi_annotations
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

module load python/3.8.0

/home/hyruuk/python_envs/shinobi_env/bin/python /project/rrg-pbellec/hyruuk/shinobi_fmri/data/shinobi_behav/shinobi_behav/annotations/generate_annotations.py
