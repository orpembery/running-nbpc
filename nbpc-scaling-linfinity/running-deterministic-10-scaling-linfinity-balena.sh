#!/bin/bash

source /beegfs/scratch/user/s/orp20/own-utilities/helmholtz_firedrake_venv.sh

for k in 160.0 #20.0 40.0 60.0 80.0 100.0 120.0 140.0
do
    for beta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
	num_procs=$(python calc_procs.py $k)

	num_nodes=$(python calc_nodes.py $num_procs)
	
	sbatch --nodes=$num_nodes ./jobscript-nbpc-deterministic.slurm $num_procs $k 0.5 $beta 0 1 10
    done
done
