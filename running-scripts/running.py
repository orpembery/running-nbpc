from helmholtz_nearby_preconditioning import experiments
from firedrake import petsc
# I hope the following is hack that will mean the mesh generation errors don't happen.

import sys

k = float(sys.argv[1])

import ctypes

import os

petsc_dir = os.environ.get('PETSC_DIR', None)

petsc_arch = os.environ.get('PETSC_ARCH', None)

libpetsc_path = os.path.join(petsc_dir, petsc_arch, 'lib', 'libpetsc.so')

petsc = ctypes.CDLL(libpetsc_path)

# Do not check validity of address before dereferencing pointers

petsc.PetscCheckPointerSetIntensity(0)

import cProfile

pr = cProfile.Profile()

pr.enable()

# For A changing - Euan
#experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type='constant',n_pre_type='constant',dim=2,num_pieces=10,seed=1,num_repeats=20,k_list=[10.0,20.0,30.0,40.0],h_list=[(1.0,-1.5)],p_list=[1],noise_master_level_list=[(0.1,0.0)],noise_modifier_list=[(0.0,0.0,0.0,0.0),(0.0,-0.5,0.0,0.0),(0.0,-1.0,0.0,0.0)],save_location='./')

# For n changing - Euan
#experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type='constant',n_pre_type='constant',dim=2,num_pieces=10,seed=1,num_repeats=20,k_list=[10.0,20.0,30.0,40.0],h_list=[(1.0,-1.5)],p_list=[1],noise_master_level_list=[(0.0,0.1)],noise_modifier_list=[(0.0,0.0,0.0,0.0),(0.0,0.0,0.0,-0.5),(0.0,0.0,0.0,-1.0)],save_location='./')


# For n changing - Owen
experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type='constant',n_pre_type='constant',dim=2,num_pieces=10,seed=1,num_repeats=20,k_list=[k],h_list=[(1.0,-1.5)],p_list=[1],noise_master_level_list=[(0.0,0.1)],noise_modifier_list=[(0.0,0.0,0.0,0.0),(0.0,0.0,0.0,-0.5),(0.0,0.0,0.0,-1.0)],save_location='./')

experiments.nearby_preconditioning_piecewise_experiment_set(A_pre_type='constant',n_pre_type='constant',dim=2,num_pieces=10,seed=1,num_repeats=20,k_list=[k],h_list=[(1.0,-1.5)],p_list=[1],noise_master_level_list=[(0.0,0.05),(0.0,0.01)],noise_modifier_list=[(0.0,0.0,0.0,0.0)],save_location='./')


