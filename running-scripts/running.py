from helmholtz_nearby_preconditioning import experiments
# For A changing
#experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type='constant',n_pre_type='constant',dim=2,num_pieces=10,seed=1,num_repeats=20,k_list=[10.0,20.0,30.0,40.0],h_list=[(1.0,-1.5)],p_list=[1],noise_master_level_list=[(0.1,0.0)],noise_modifier_list=[(0.0,0.0,0.0,0.0),(0.0,-0.5,0.0,0.0),(0.0,-1.0,0.0,0.0)],save_location='./')

# For n changing
experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type='constant',n_pre_type='constant',dim=2,num_pieces=10,seed=1,num_repeats=20,k_list=[10.0,20.0,30.0,40.0],h_list=[(1.0,-1.5)],p_list=[1],noise_master_level_list=[(0.0,0.1)],noise_modifier_list=[(0.0,0.0,0.0,0.0),(0.0,0.0,0.0,-0.5),(0.0,0.0,0.0,-1.0)],save_location='./')


