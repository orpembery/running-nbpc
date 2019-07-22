from helmholtz_firedrake.problems import HelmholtzProblem
import firedrake as fd
from helmholtz_firedrake.utils import h_to_num_cells, nd_indicator
import numpy as np
import sys

on_balena = bool(int(sys.argv[1]))

print(on_balena,flush=True)

if on_balena:

    from firedrake_complex_hacks import balena_hacks
    balena_hacks.fix_mesh_generation_time()
    print('imported',flush=True)


k_list = [200.0]#[10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]#[10.0,20.0,30.0,40.0,50.0,60.0]

#eps_list = [0.1,0.01,0.001]

# What if you took eps = 1/k

discon = np.array([[0.5,1.0],[0.0,1.0]])

angle = 2.0*np.pi/3.0

eps_const = 0.2

for k in k_list:

    for eps_power in [0.9]:#,0.8,0.7,0.6]:

        eps = eps_const/k**eps_power

        shift = np.array([[eps,0.0],[0.0,0.0]])

        num_cells = h_to_num_cells(k**(-1.5),2)

        mesh = fd.UnitSquareMesh(num_cells,num_cells)

        V = fd.FunctionSpace(mesh,"CG",1)

        x = fd.SpatialCoordinate(mesh)

        n = 0.5 + nd_indicator(x,1.0,discon+eps)

        n_pre = 0.5 + nd_indicator(x,1.0,discon)

        A = fd.as_matrix([[1.0,0.0],[0.0,1.0]])

        prob = HelmholtzProblem(k,V,A=A,n=n,A_pre=A,n_pre=n_pre)

        prob.f_g_plane_wave([np.cos(angle),np.sin(angle)])

        prob.solve()

        if fd.COMM_WORLD.rank == 0:

            print('k',k,'eps_const',eps_const,'eps_power',eps_power,'eps',eps,'GMRES iterations',prob.GMRES_its,flush=True)





