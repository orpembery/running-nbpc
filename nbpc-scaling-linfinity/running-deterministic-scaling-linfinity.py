import sys
import helmholtz_firedrake.problems as hh
import helmholtz_firedrake.utils as hh_utils
import firedrake as fd
import numpy as np

# This is a slightly rough-and-ready experiment to assess whether just taking the worst case on an alternating checkerboard gives substantially different numbers of GMRES iterations to those already in the paper.
# It borrows from other code, but has had to be reworked for the deterministic scenario
# Many input arguments: k, alpha, beta, bool (1 if want to do A computation, 0 if n),  bool: 1 if on balena, 0 otherwise

on_balena = bool(int(sys.argv[5]))

if on_balena:
    from firedrake_complex_hacks.balena_hacks import fix_mesh_generation_time
    fix_mesh_generation_time()

k = float(sys.argv[1])

alpha = float(sys.argv[2])

beta = float(sys.argv[3])

k_multipler = k**(-beta)

A_vs_n = bool(int(sys.argv[4]))

dim = 2

h = k**(-1.5)

mesh_points = hh_utils.h_to_num_cells(h,dim)
mesh = fd.UnitSquareMesh(mesh_points,mesh_points)
V = fd.FunctionSpace(mesh, "CG", 1)

f = 0.0
d = fd.as_vector([1.0/fd.sqrt(2.0),1.0/fd.sqrt(2.0)])
x = fd.SpatialCoordinate(mesh)
nu = fd.FacetNormal(mesh)
g = 1j*k*fd.exp(1j*k*fd.dot(x,d))*(fd.dot(d,nu)-1)

if A_vs_n:
    constant_to_multiply = fd.as_matrix([[1.0,0.0],[0.0,1.0]])
    varying_coeff = fd.as_matrix([[1.0,0.0],[0.0,1.0]])
else:
    constant_to_multiply = 1.0
    varying_coeff = 1.0

# This is, strictly speaking, the number of subdomains in each direction
num_pieces = 10

for ii in range(num_pieces):
    for jj in range(num_pieces):
        if np.mod(ii+jj,2) == 1:
            value = + alpha * k_multipler
        else:
            value = - alpha* k_multipler
        fl_ii = float(ii)
        fl_jj = float(jj)
        print(ii,jj,value)
        varying_coeff += hh_utils.nd_indicator(x,value * constant_to_multiply,np.array([[fl_ii,fl_ii+1.0],[fl_jj,fl_jj+1.0]])/float(num_pieces))

if A_vs_n:        
    A = varying_coeff
    n = 1.0
else:
    A = fd.as_matrix([[1.0,0.0],[0.0,1.0]])
    n = varying_coeff

A_pre = fd.as_matrix([[1.0,0.0],[0.0,1.0]])

n_pre = 1.0

prob = hh.HelmholtzProblem(k=k, V=V, A=A, n=n, A_pre=A_pre, n_pre=n_pre)



prob.solve()

GMRES_its = []

if fd.COMM_WORLD.rank == 0:
        
    GMRES_its.append(prob.GMRES_its)

GMRES_its  = np.array(GMRES_its)

# Write this to file

save_location = './output-deterministic-10/'

h_tuple = [1.0,-1.5]

p = 1

A_pre_type='constant'

n_pre_type='constant'

dim = 2
if A_vs_n:
    noise_master = (alpha,0.0)
else:
    noise_master = (0.0,alpha)

if A_vs_n:
    modifier = (0.0,-beta,0.0,0.0)
else:
    modifier = (0.0,.0,-0.0,-beta)

num_repeats = 1

if fd.COMM_WORLD.rank == 0:
    hh_utils.write_GMRES_its(
        GMRES_its,save_location,
        {'k' : k,
         'h_tuple' : h_tuple,
         'p' : p,
         'num_pieces' : num_pieces,
         'A_pre_type' : A_pre_type,
         'n_pre_type' : n_pre_type,
         'noise_master' : noise_master,
         'modifier' : modifier,
         'num_repeats' : num_repeats
         }
    )

    


