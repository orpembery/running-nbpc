#!/bin/bash

# k is argument 1 (include decimal point)

# beta is argument 2

# checkerboard (2 or 10) is argument 3

source /beegfs/scratch/user/s/orp20/own-utilities/helmholtz_firedrake_venv.sh

num_procs=$(python calc_procs.py $1)

num_nodes=$(python calc_nodes.py $num_procs)
	
sbatch --nodes=$num_nodes ./jobscript-nbpc-deterministic.slurm $num_procs $1 0.5 $2 0 1 $3