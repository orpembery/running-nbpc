from helmholtz_nearby_preconditioning import experiments
import sys

# First argument should be k
# Second argument should be 1 if on balena, 0 otherwise

on_balena = bool(int(sys.argv[2]))

if on_balena:
    from firedrake_complex_hacks.balena_hacks import fix_mesh_generation_time
    fix_mesh_generation_time()

k_list = [float(sys.argv[1])]

h_list = [(1.0,-1.5)]

p_list = [1]

num_repeats = 100

num_pieces = 10

seed = 1

A_pre_type='constant'

n_pre_type='constant'

dim = 2

save_location = './output/'

# For A changing
experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type=A_pre_type,n_pre_type=n_pre_type,dim=dim,num_pieces=num_pieces,seed=seed,num_repeats=num_repeats,k_list=k_list,h_list=h_list,p_list=p_list,noise_master_level_list=[(0.1,0.0)],noise_modifier_list=[(0.0,0.0,0.0,0.0),(0.0,-0.5,0.0,0.0),(0.0,-1.0,0.0,0.0)],save_location=save_location)

# For n changing
experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type=A_pre_type,n_pre_type=n_pre_type,dim=dim,num_pieces=num_pieces,seed=seed,num_repeats=num_repeats,k_list=k_list,h_list=h_list,p_list=p_list,noise_master_level_list=[(0.0,0.1)],noise_modifier_list=[(0.0,0.0,0.0,0.0),(0.0,0.0,0.0,-0.5),(0.0,0.0,0.0,-1.0)],save_location=save_location)


