#!/bin/bash

#SBATCH --account=rc-ma1200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=batch
#SBATCH --time=96:00:00

source /beegfs/scratch/user/s/orp20/own-utilities/helmholtz_firedrake_venv.sh

cd /home/s/orp20/scratch/running-nbpc/nbpc-scaling-linfinity

mpirun -n $1 python running-nbpc-scaling-linfinity.py $2 $3 $4 $5
