import helmholtz_nearby_preconditioning.experiments as nbex
from sys import argv

k = 5.0

nbex.nearby_preconditioning_piecewise_experiment_set(
        'constant','constant',10,2,1000,[k],[(1.0,-1.5)],[(0.0,0.1),(0.01,0.0)],
        [(0.0,0.0,0.0,0.0),(0.0,-1.0,0.0,-1.0)],
        '/home/s/orp20/scratch/helmholtz-nearby-preconditioning/dev/')
