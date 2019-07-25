#!/bin/bash

source /beegfs/scratch/user/s/orp20/own-utilities/helmholtz_firedrake_venv.sh

for k in 20.0 40.0 60.0 80.0 100.0
do
    num_procs=$(python calc_procs.py $k)
    
    sbatch --job-name="$k" --err="$k.err" --output="$k.out" ./jobscript-nbpc.slurm $num_procs $k 1
done
