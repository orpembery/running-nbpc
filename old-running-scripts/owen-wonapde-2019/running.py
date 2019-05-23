from helmholtz_nearby_preconditioning import experiments
# I hope the following is hack that will mean the mesh generation errors don't happen.

#import sys

#k = float(sys.argv[1])

k_list = [20.0,40.0,60.0,80.0]

# For A changing
experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type='constant',n_pre_type='constant',dim=2,num_pieces=10,seed=1,num_repeats=20,k_list=k_list,h_list=[(1.0,-1.5)],p_list=[1],noise_master_level_list=[(0.05,0.0)],noise_modifier_list=[(0.0,0.0,0.0,0.0),(0.0,-0.5,0.0,0.0),(0.0,-1.0,0.0,0.0)],save_location='./')

# For n changing - Owen
#experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type='constant',n_pre_type='constant',dim=2,num_pieces=10,seed=1,num_repeats=20,k_list=k_list,h_list=[(1.0,-1.5)],p_list=[1],noise_master_level_list=[(0.0,0.1)],noise_modifier_list=[(0.0,0.0,0.0,0.0),(0.0,0.0,0.0,-0.5),(0.0,0.0,0.0,-1.0)],save_location='./')

