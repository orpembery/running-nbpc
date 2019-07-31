#!/bin/bash

#SBATCH --account=rc-ma1200
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=batch
#SBATCH --time=48:00:00

source /beegfs/scratch/user/s/orp20/own-utilities/helmholtz_firedrake_venv.sh

cd /home/s/orp20/scratch/running-nbpc/nbpc-scaling-l1

mpirun -n 64 python running-nbpc-scaling-l1.py 1
