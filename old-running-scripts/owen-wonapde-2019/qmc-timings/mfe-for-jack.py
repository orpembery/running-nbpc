from firedrake import *

mesh = UnitSquareMesh(100,100)

V = FunctionSpace(mesh,"CG",1)

n_pre = Constant(1.0)

n = Constant(1.0)

u = TrialFunction(V)

v = TestFunction(V)

a = inner(n * u,v)*dx

a_pre = inner(n_pre * u,v)*dx

L = inner(1.0,v)*dx

u_h = Function(V)

problem = LinearVariationalProblem(a, L, u_h,aP=a_pre, constant_jacobian=False)
       
solver = LinearVariationalSolver(problem, solver_parameters =
                                    {"ksp_type": "gmres",
                                     "mat_type": "aij",
                                     "snes_lag_preconditioner": -1,
                                     "ksp_reuse_preconditioner": True,
                                     "pmat_type": "aij",
                                     "pc_type": "lu",
                                     "ksp_norm_type": "unpreconditioned"})

for ii in range(20):

    solver.solve()
