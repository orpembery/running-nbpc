from helmholtz_firedrake.problems import HelmholtzProblem
from firedrake import *

k = 10.0

num_points = 2 * int(k**1.5)

mesh = UnitSquareMesh(num_points,num_points)

V = FunctionSpace(mesh,"CG",1)

prob = HelmholtzProblem(k,V)

prob.set_A_pre(prob._A)

prob.set_n_pre(prob._n)

prob.solve()

print(prob.GMRES_its)
