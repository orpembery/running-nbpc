from helmholtz_nearby_preconditioning.experiments import qmc_nbpc_experiment
from running_helmholtz_monte_carlo.calculate_m import calc_M
import numpy as np
import pandas
import subprocess
import datetime
import firedrake as fd
import sys
# This script runs Quasi-Monte-Carlo for the Helmholtz IIP, with nearby preconditioning used to speed up the solves.
# It also scales the number of QMC points with k, so (we hope) the QMC error is bounded as k increases.
# First command-line argument is k

# NOTE, the values in here are based on one particular set of analyses for how QMC behaves. IF you want to run for a different set of analyses, you (currently) must change things manually!

k_list = [float(sys.argv[1])]

h_spec = (1.0,-1.5)

dim = 2

J = 10

delta = 1.0

j_scaling = 4.0

lambda_mult = 1.0

mean_type = 'constant'

use_nbpc = True

points_generation_method = 'qmc'

seed = 1

GMRES_threshold = np.Inf

for k in k_list:

    M = calc_M(k)

    points_info = qmc_nbpc_experiment(h_spec,dim,J,M,k,delta,lambda_mult,j_scaling,mean_type,
                        use_nbpc,points_generation_method,seed,GMRES_threshold)

    if fd.COMM_WORLD.rank ==0:
        # Get current date and time
        # This initialises the object. I don't understand why this is
        # necessary.
        date_time = datetime.datetime(1,1,1)

        # Get git hash
        git_hash = subprocess.run("git rev-parse HEAD", shell=True,
                                  stdout=subprocess.PIPE)
        
        # help from https://stackoverflow.com/a/6273618
        file_name = 'mean-based-k-'+str(k)+'-git-hash-' + git_hash.stdout.decode('UTF-8')[:-1][:6]+date_time.utcnow().isoformat()+'.csv'

        points_info.to_csv(file_name)
        
