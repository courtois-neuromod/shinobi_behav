for LEVEL in 1-0 4-1 5-0; do
  for SUBJECT in 01 02 04 06; do
    sbatch slurm/subm_make_dataset.sh $SUBJECT $LEVEL
done
done
