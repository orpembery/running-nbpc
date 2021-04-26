from helmholtz_nearby_preconditioning import experiments
import sys

# First argument should be k
# Second argument should be 1 if on balena, 0 otherwise
# Third argument should be which 'set' of beta you want to run
# 0 --> beta = 0.0
# 1 --> beta = [-0.1,-0.2]
# 2 --> beta = [-0.3,-0.4,-0.5]
# 3 --> beta = [-0.6,-0.7,-0.8,-0.9,-1.0]
# Fourth argument should be number of repeats


on_balena = bool(int(sys.argv[2]))

if on_balena:
    from firedrake_complex_hacks.balena_hacks import fix_mesh_generation_time
    fix_mesh_generation_time()

beta_set = int(sys.argv[3])

k_list = [float(sys.argv[1])]

h_list = [(1.0,-1.5)]

p_list = [1]

num_repeats = int(sys.argv[4])

num_pieces = 10

seed = 1

A_pre_type='constant'

n_pre_type='constant'

dim = 2

if num_repeats == 100:
    save_location = './output/'
elif num_repeats == 200:
    save_location = './output-200-repeats/'

noise_master_level = 0.5


# For A changing
A_noise_modifiers = [[(0.0,0.0,0.0,0.0)],[(0.0,-0.1,0.0,0.0),(0.0,-0.2,0.0,0.0)],[(0.0,-0.3,0.0,0.0),(0.0,-0.4,0.0,0.0),(0.0,-0.5,0.0,0.0)],[(0.0,-0.6,0.0,0.0),(0.0,-0.7,0.0,0.0),(0.0,-0.8,0.0,0.0),(0.0,-0.9,0.0,0.0),(0.0,-1.0,0.0,0.0)]]

experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type=A_pre_type,n_pre_type=n_pre_type,dim=dim,num_pieces=num_pieces,seed=seed,num_repeats=num_repeats,k_list=k_list,h_list=h_list,p_list=p_list,noise_master_level_list=[(noise_master_level,0.0)],noise_modifier_list=A_noise_modifiers[beta_set],save_location=save_location)

print("A complete")

# For n changing
n_noise_modifiers = [[(0.0,0.0,0.0,0.0)],[(0.0,0.0,0.0,-0.1),(0.0,0.0,0.0,-0.2)],[(0.0,0.0,0.0,-0.3),(0.0,0.0,0.0,-0.4),(0.0,0.0,0.0,-0.5)],[(0.0,0.0,0.0,-0.6),(0.0,0.0,0.0,-0.7),(0.0,0.0,0.0,-0.8),(0.0,0.0,0.0,-0.9),(0.0,0.0,0.0,-1.0)]]

experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type=A_pre_type,n_pre_type=n_pre_type,dim=dim,num_pieces=num_pieces,seed=seed,num_repeats=num_repeats,k_list=k_list,h_list=h_list,p_list=p_list,noise_master_level_list=[(0.0,noise_master_level)],noise_modifier_list=n_noise_modifiers[beta_set],save_location=save_location)


print("n complete")
