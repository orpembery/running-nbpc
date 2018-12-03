import helmholtz_nearby_preconditioning.experiments as nbex

nbex.nearby_preconditioning_piecewise_experiment_set(
        'constant','constant',2,1,2,[10.0],[(1.0,-1.0)],[(0.01,0.1)],
        [(0.0,0.0,0.0,0.0)],
        '/home/owen/code/helmholtz-nearby-preconditioning/dev/')
