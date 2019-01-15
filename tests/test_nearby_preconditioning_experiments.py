import helmholtz_firedrake.problems as hh
import firedrake as fd
import numpy as np
import helmholtz_nearby_preconditioning.experiments as nbex
from helmholtz_firedrake.coefficients import PiecewiseConstantCoeffGenerator
import helmholtz_firedrake.utils as hh_utils
import copy
from os import mkdir
from shutil import rmtree
import latticeseq_b2

def test_coeff_definition_error():
    """Test that a coeff with too many pieces is caught."""
    k = 20.0
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 13
    noise_level = 0.1
    num_repeats = 10
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    n_pre = 1.0
    n_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,n_pre,[1])

    f = 1.0
    g = 1.0
    
    GMRES_its = nbex.nearby_preconditioning_experiment(V,k,A_pre,A_stoch,
                                n_pre,n_stoch,f,g,num_repeats)

    # The code should catch the error and print a warning message, and
    # exit, not recording any GMRES iterations.

    assert GMRES_its.shape == (0,)

def test_coeff_definition_no_error():
    """Test that a coeff with just too few many pieces is not caught."""
    k = 20.0
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 12
    noise_level = 0.1
    num_repeats = 10
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    n_pre = 1.0
    n_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,n_pre,[1])

    f = 1.0
    g = 1.0
    
    GMRES_its = nbex.nearby_preconditioning_experiment(V,k,A_pre,A_stoch,
                                n_pre,n_stoch,f,g,num_repeats)

    # The code should not error.

    assert GMRES_its.shape != (0,)


def test_nearby_preconditioning_set_no_errors():
    """Tests that a basic run of a set doesn't produce any errors."""

    mkdir('./tmp')
    
    nbex.nearby_preconditioning_piecewise_experiment_set(
        'constant','constant',2,2,1,2,[10.0],[(1.0,-1.0)],[1],[(0.01,0.1)],
        [(0.0,0.0,0.0,0.0)],
        './tmp/'
    )

    rmtree('./tmp')

def test_nearby_preconditioning_jump_up_set_no_errors():
    """Tests a run with n jumping up produces no errors."""

    mkdir('./tmp')

    nbex.nearby_preconditioning_piecewise_experiment_set(
        'constant','jump_up',2,2,1,2,[10.0],[(1.0,-1.0)],[1],[(0.01,0.1)],
        [(0.0,0.0,0.0,0.0)],
        './tmp/'
    )

    rmtree('./tmp')

def test_nearby_preconditioning_jump_down_set_no_errors():
    """Tests a run with n jumping down produces no errors."""

    mkdir('./tmp')

    nbex.nearby_preconditioning_piecewise_experiment_set(
        'constant','jump_down',2,2,1,2,[10.0],[(1.0,-1.0)],[1],[(0.01,0.1)],
        [(0.0,0.0,0.0,0.0)],
        './tmp/'
    )

    rmtree('./tmp')

def test_nearby_preconditioning_set_higher_p_no_errors():
    """Tests that a set with higher p doesn't produce any errors."""

    mkdir('./tmp')
    
    nbex.nearby_preconditioning_piecewise_experiment_set(
        'constant','constant',2,2,1,2,[10.0],[(1.0,-1.0)],[2],[(0.01,0.1)],
        [(0.0,0.0,0.0,0.0)],
        './tmp/'
    )

    rmtree('./tmp')

def test_qmc_works():
    """This just tests that the code runs."""

    np.random.seed(7)
    
    h_spec = (1.0,-1.0)

    dim = 2

    J = 4

    M = 4

    k = 10.0

    delta = 1.0

    lambda_mult = 1.0

    mean_type = 'constant'

    use_nbpc = True

    GMRES_threshold = 10
    
    nbex.qmc_nbpc_experiment(h_spec,dim,J,M,k,delta,lambda_mult,
                             mean_type,use_nbpc,GMRES_threshold)

def test_qmc_no_nbpc_working():
    """Tests that the nearby preconditioning strategy is being done."""

    np.random.seed(4)

    h_spec = (1.0,-1.5)

    dim = 2

    J = 10

    M = 5

    k = 10.0

    delta = 1.0

    lambda_mult = 1.0

    mean_type = 'constant'

    use_nbpc = False

    GMRES_threshold = 10
    
    points_info = nbex.qmc_nbpc_experiment(h_spec,dim,J,M,k,delta,lambda_mult,
                                           mean_type,use_nbpc,GMRES_threshold)

    # Correct number of points
    assert points_info.shape[0] == 2**M
    
    # Every point is an expected QMC point - generate again using Dirk
    # Nuyen's code
    qmc_generator = latticeseq_b2.latticeseq_b2(s=J)

    points = []
    
    for m in range((M+1)):
        points.append(qmc_generator.calc_block(m))

    qmc_points = points[0]

    for ii in range(1,len(points)):
        qmc_points = np.vstack((qmc_points,points[ii]))

    qmc_points -= 0.5

    for ii in range(2**M):
        assert (points_info.loc[ii,"sto_loc"] == qmc_points).all(axis=1).any()
    
    # We performed an LU decomposition for every point
    assert all(points_info.loc[:,"LU"])

    # All GMRES iterations are 1
    assert all(points_info.GMRES == 1)

def test_qmc_nbpc_working():
    """Tests that the nearby preconditioning strategy is being done."""

    np.random.seed(4)

    h_spec = (1.0,-1.5)

    dim = 2

    J = 10

    M = 5

    k = 10.0

    delta = 1.0

    lambda_mult = 1.0

    mean_type = 'constant'

    use_nbpc = True

    GMRES_threshold = 10
    
    points_info = nbex.qmc_nbpc_experiment(h_spec,dim,J,M,k,delta,lambda_mult,
                                           mean_type,use_nbpc,GMRES_threshold)

    # Correct number of points
    assert points_info.shape[0] == 2**M
    
    # Every point is an expected QMC point - generate again using Dirk
    # Nuyen's code
    qmc_generator = latticeseq_b2.latticeseq_b2(s=J)

    points = []
    
    for m in range((M+1)):
        points.append(qmc_generator.calc_block(m))

    qmc_points = points[0]

    for ii in range(1,len(points)):
        qmc_points = np.vstack((qmc_points,points[ii]))

    qmc_points -= 0.5

    for ii in range(2**M):
        assert (points_info.loc[ii,"sto_loc"] == qmc_points).all(axis=1).any()

    
    # If we performed an LU decomposition, GMRES converged in one step
    assert all(points_info.loc[lambda df: df.LU == True].GMRES == 1)

    # All GMRES iterations are less than the threshold
    assert all(points_info.GMRES <= GMRES_threshold)

