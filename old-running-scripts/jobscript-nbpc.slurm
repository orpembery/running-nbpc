#!/bin/bash

#SBATCH --account=rc-ma1200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=batch
#SBATCH --time=06:00:00

source /beegfs/scratch/user/s/orp20/own-utilities/helmholtz_firedrake_venv.sh

cd /home/s/orp20/scratch/helmholtz-nearby-preconditioning/running-scripts

mpirun -n $1 python running.py $2