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


def test_nearby_preconditioning_set_no_errors():
    """Tests that a basic run of a set doesn't produce any errors."""

    nbex.nearby_preconditioning_piecewise_experiment_set(
        'constant','constant',2,1,2,[10.0],[(1.0,-1.0)],[(0.01,0.1)],
        [(0.0,0.0,0.0,0.0)],
        '/home/owen/code/helmholtz-firedrake/testing-output/for-pytest/'
    )

