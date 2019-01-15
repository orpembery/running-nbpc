import helmholtz_nearby_preconditioning.experiments as nbex

h_spec = (1.0,-1.5)

dim = 2

J = 4

M = 4

k = 10.0

delta = 1

lambda_mult = 1

mean_type = 'constant'

use_nbpc = True

GMRES_threshold = 10

nbex.qmc_nbpc_experiment(h_spec,dim,J,M,k,delta,lambda_mult,mean_type,use_nbpc,GMRES_threshold)


