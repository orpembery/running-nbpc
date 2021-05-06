#!/bin/bash

source /beegfs/scratch/user/s/orp20/own-utilities/helmholtz_firedrake_venv.sh

for k in 20.0 40.0 60.0 80.0 100.0
do
for betas in 0 1 2 3
do
for reps in 100 200
do
num_procs=$(python calc_procs.py $k)
   
sbatch ./jobscript-nbpc.slurm $num_procs $k 1 $betas $reps
done
done
done
