from helmholtz_firedrake.problems import HelmholtzProblem
import firedrake as fd
from helmholtz_firedrake.utils import h_to_num_cells, nd_indicator, write_repeats_to_csv
import numpy as np
import sys
import pandas as pd

# Command line argument should be 1 if on Balena, 0 otherwise

on_balena = bool(int(sys.argv[1]))

if on_balena:
    from firedrake_complex_hacks import balena_hacks
    balena_hacks.fix_mesh_generation_time()

k_list = np.linspace(10.0,150.0,num=10)

discon = np.array([[0.5,1.0],[0.0,1.0]])

angle = 2.0*np.pi/3.0

eps_const = 0.2

for eps_power in np.linspace(0.0,1.0,num=11):

    storage = np.ones((1,2))

    print(eps_power,flush=True)
    
    for k in k_list:

        print(k,flush=True)

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
        
        storage=np.append(storage,np.array((k,prob.GMRES_its),ndmin=2),axis=0)

    storage = storage[1:,:]

    if fd.COMM_WORLD.rank == 0:
        
        write_repeats_to_csv(storage,'./output/','l1-scaling-power-'+str(eps_power),{})





