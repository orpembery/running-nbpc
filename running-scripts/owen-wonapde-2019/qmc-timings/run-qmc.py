import helmholtz_nearby_preconditioning.experiments as nbex
from pyop2.profiling import timed_stage

h_spec = (1.0,-1.5)

dim = 2

J = 4

M = 5

k = 10.0

delta = 1

lambda_mult = 1

mean_type = 'constant'

GMRES_threshold = 10


with timed_stage("No nbpc"):
    use_nbpc = False
    points_info = nbex.qmc_nbpc_experiment(h_spec,dim,J,M,k,delta,lambda_mult,mean_type,use_nbpc,GMRES_threshold)

with timed_stage("nbpc"):
    use_nbpc = True
    points_info = nbex.qmc_nbpc_experiment(h_spec,dim,J,M,k,delta,lambda_mult,mean_type,use_nbpc,GMRES_threshold)

    



