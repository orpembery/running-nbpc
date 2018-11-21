import helmholtz.problems as hh
import firedrake as fd
import numpy as np
import nearby_preconditioning.experiments as nbex
from helmholtz.coefficients import PiecewiseConstantCoeffGenerator
import helmholtz.utils as hh_utils
import copy

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

def test_coeff_being_updated():
    """Test that the random coefficients are actually updated."""

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

    A_copy = copy.deepcopy(A_stoch._coeff_values[0].values())

    n_copy = copy.deepcopy(n_stoch._coeff_values[0].values())

    A_stoch.sample()

    n_stoch.sample()

    A_diff = A_copy - A_stoch._coeff_values[0].values()

    assert all(A_copy != 0.0)

    assert n_copy != n_stoch._coeff_values[0].values()

def test_h_to_mesh_points():
    """Test that given h, the correct number of points is calculated."""

    for h_power in range(10):
        h = 2**(-float(h_power))

        assert hh_utils.h_to_mesh_points(h) == np.ceil(np.sqrt(2.0)/h)
    
    
def test_coeff_size():
    """Tests that the coeffs generated are the correct size."""

    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 1
    noise_level = 0.1
    num_repeats = 100
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    n_pre = 1.0
    n_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,n_pre,[1])
    for ii in range(num_repeats):
        A_stoch.sample()
        n_stoch.sample()
        for jj in range(num_pieces**2):
            assert A_stoch._coeff_values[jj].evaluate(None,None,(),None).shape\
                == (2,2)
            assert n_stoch._coeff_values[jj].evaluate(None,None,(),None).shape\
                == ()
    
def test_matrices_spd():
    """Tests that the matrices are spd, using Sylvester's criterion.

    Only works for the case coeff_pre = I."""

    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 12
    noise_level = 0.1
    num_repeats = 100
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    for ii in range(num_repeats):
        A_stoch.sample()

        for jj in range(num_pieces**2):

            assert A_stoch._coeff_values[jj].evaluate(None,None,(),None)[1,0]\
                == A_stoch._coeff_values[jj].evaluate(None,None,(),None)[0,1]
            
            assert 1.0 +\
                A_stoch._coeff_values[jj].evaluate(None,None,(),None)[0,0]\
                > 0.0
            
            assert np.linalg.det(np.array([[1.0,0.0],[0.0,1.0]])\
                                 + A_stoch._coeff_values[jj].\
                                 evaluate(None,None,(),None))\
                                 > 0.0

def test_matrices_noise_level():
    """Tests that the matrices have correct noise_level.

    Only works for the case coeff_pre = I."""

    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 12
    noise_level = 0.1
    num_repeats = 100
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    for ii in range(num_repeats):
        A_stoch.sample()

        for jj in range(num_pieces**2):

            assert abs(A_stoch._coeff_values[jj]\
                       .evaluate(None,None,(),None)[0,0]) <= noise_level

            assert abs(A_stoch._coeff_values[jj]\
                       .evaluate(None,None,(),None)[1,1]) <= noise_level

            assert abs(A_stoch._coeff_values[jj]\
                       .evaluate(None,None,(),None)[0,1]) <= noise_level

            assert abs(A_stoch._coeff_values[jj]\
                       .evaluate(None,None,(),None)[1,0]) <= noise_level

def test_nearby_preconditioning_set_no_errors():
    """Tests that a basic run of a set doesn't produce any errors."""

    nbex.nearby_preconditioning_piecewise_experiment_set(
        'constant','constant',2,1,2,[10.0],[(1.0,-1.0)],[(0.01,0.1)],
        [(0.0,0.0,0.0,0.0)],
        '/home/owen/code/helmholtz-firedrake/testing-output/for-pytest/'
    )

