#!/bin/bash

for k in 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0
do
    sbatch --job-name="$k" --err="$k" --output="$k" ./jobscript.slurm $ki 
done
