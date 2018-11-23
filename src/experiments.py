import firedrake as fd
import helmholtz.problems as hh
import helmholtz.coefficients as coeff
import helmholtz.utils as hh_utils
import numpy as np
import pandas as pd
import firedrake_complex_compute_error as error
from shutil import move
from matplotlib import pyplot as plt
from matplotlib import cm

def nearby_preconditioning_experiment(V,k,A_pre,A_stoch,n_pre,n_stoch,f,g,
                                num_repeats):
    """For a given preconditioning Helmholtz problem, performs a test of
    the effectiveness of nearby preconditioning.

    For a given preconditioning Helmholtz problem, and given methods for
    generating realisations of Helmholtz problems with random field
    coefficients, generates realisations of the Helmholtz problems, and
    then uses the preconditioner to perform preconditioned GMRES. Then
    records the number of GMRES iterations needed to acheive
    convergence.

    Parameters:

    V - see HelmholtzProblem

    k - see HelmholtzProblem

    A_pre - see HelmholtzProblem

    A_stoch - see StochasticHelmholtzProblem

    n_pre - see HelmholtzProblem

    n_stoch - see StochasticHelmholtzProblem

    f - see HelmholtzProblem

    g - see HelmholtzProblem

    num_repeats - int, specifying the number of realisations to take.


    Returns: numpy array of ints of length num_repeats, giving the
    number of GMRES iterations for the different realisations.
    """

    prob = hh.StochasticHelmholtzProblem(
        k=k, V=V, A_stoch=A_stoch, n_stoch=n_stoch,
        **{"A_pre": A_pre, "n_pre" : n_pre, "f" : f, "g" : g})

    all_GMRES_its = []

    for ii_repeat in range(num_repeats):
        print(ii_repeat)
        try:
            prob.solve()
        except RecursionError:
            print("Suffered a Python RecursionError.\
            Have you specified something using a big loop in UFL?\
            Aborting all further solves.")
            break
            
        all_GMRES_its.append(prob.GMRES_its)

        prob.sample()

    all_GMRES_its  = np.array(all_GMRES_its)

    return all_GMRES_its

def nearby_preconditioning_piecewise_experiment_set(
        A_pre_type,n_pre_type,num_pieces,seed,num_repeats,
        k_list,h_list,noise_master_level_list,noise_modifier_list,
        save_location):
    """Test nearby preconditioning for a range of parameter values.

    Performs nearby preconditioning tests for a range of values of k,
    the mesh size h, and the size of the random noise (which can be
    specified in terms of k and h). The random noise is piecewise
    constant on a grid unrelated to the finite-element mesh.

    Parameters:

    A_pre_type - string - options are 'constant', giving A_pre =
    [[1.0,0.0],[0.0,1.0]].

    n_pre_type - string - options are 'constant', giving n_pre = 1.0.

    num_pieces - see
        helmholtz.coefficients.PieceWiseConstantCoeffGenerator.

    seed - see StochasticHelmholtzProblem.

    num_repeats - see nearby_preconditioning_test.

    k_list - list of positive floats - the values of k for which we will
    run experiments.

    h_list - list of 2-tuples; in each tuple (call it t) t[0] should be
    a positive float and t[1] should be a float. These specify the
    values of the mesh size h for which we will run experiments. h =
    t[0] * k**t[1].

    noise_master_level_list - list of 2-tuples, where each entry of the
    tuple is a positive float.  This defines the values of base_noise_A
    and base_noise_n to be used in the experiments. Call a given tuple
    t. Then base_noise_A = t[0] and base_noise_n = t[1].

    noise_modifier_list - list of 4-tuples; the entries of each tuple
    should be floats. Call a given tuple t. This modifies the base noise
    so that the L^\infty norms of A and n are less than or equal to
    (respectively) base_noise_A * h**t[0] * k**t[1] and base_noise_n *
    h**t[2] * k**t[3].

    save_location - see utils.write_repeats_to_csv.
    """

    if not(isinstance(A_pre_type,str)):
        raise TypeError("Input A_pre_type should be a string")
    elif A_pre_type is not "constant":
        raise HelmholtzNotImplementedError(
            "Currently only implemented A_pre_type = 'constant'.")

    if not(isinstance(n_pre_type,str)):
        raise TypeError("Input n_pre_type should be a string")
    elif n_pre_type is not "constant":
        raise HelmholtzNotImplementedError(
            "Currently only implemented n_pre_type = 'constant'.")

    if not(isinstance(k_list,list)):
        raise TypeError("Input k_list should be a list.")
    elif any(not(isinstance(k,float)) for k in k_list):
        raise TypeError("Input k_list should be a list of floats.")
    elif any(k <= 0 for k in k_list):
        raise TypeError(
            "Input k_list should be a list of positive floats.")

    if not(isinstance(h_list,list)):
        raise TypeError("Input h_list should be a list.")
    elif any(not(isinstance(h_tuple,tuple)) for h_tuple in h_list):
        raise TypeError("Input h_list should be a list of tuples.")
    elif any(len(h_tuple) is not 2 for h_tuple in h_list):
        raise TypeError("Input h_list should be a list of 2-tuples.")
    elif any(not(isinstance(h_tuple[0],float)) for h_tuple in h_list)\
             or any(h_tuple[0] <= 0 for h_tuple in h_list):
        raise TypeError(
            "The first item of every tuple in h_list\
            should be a positive float.")
    elif any(not(isinstance(h_tuple[1],float)) for h_tuple in h_list):
        raise TypeError(
            "The second item of every tuple in h_list should be a float.")

    if not(isinstance(noise_master_level_list,list)):
        raise TypeError(
            "Input noise_master_level_list should be a list.")
    elif any(not(isinstance(noise_tuple,tuple))
             for noise_tuple in noise_master_level_list):
        raise TypeError(
            "Input noise_master_level_list should be a list of tuples.")
    elif any(len(noise_tuple) is not 2
             for noise_tuple in noise_master_level_list):
        raise TypeError(
            "Input noise_master_level_list should be a list of 2-tuples.")
    elif any(any(not(isinstance(noise_tuple[i],float))
                 for i in range(len(noise_tuple)))
             for noise_tuple in noise_master_level_list):
        raise TypeError(
            "Input noise_master_level_list\
            should be a list of 2-tuples of floats.")

    if not(isinstance(noise_modifier_list,list)):
        raise TypeError("Input noise_modifier_list should be a list.")
    elif any(not(isinstance(mod_tuple,tuple))
             for mod_tuple in noise_modifier_list):
        raise TypeError(
            "Input noise_modifier_list should be a list of tuples.")
    elif any(len(mod_tuple) is not 4 for mod_tuple in noise_modifier_list):
        raise TypeError(
            "Input noise_modifier_list should be a list of 4-tuples.")
    elif any(any(not(isinstance(mod_tuple[i],float))
                 for i in range(len(mod_tuple)))
             for mod_tuple in noise_modifier_list):
        raise TypeError(
            "Input noise_modifier_list\
            should be a list of 4-tuples of floats.")

    if A_pre_type is "constant":
        A_pre = fd.as_matrix([[1.0,0.0],[0.0,1.0]])

    if n_pre_type is "constant":
        n_pre = 1.0
    
    for k in k_list:
        for h_tuple in h_list:
            h = h_tuple[0] * k**h_tuple[1]
            mesh_points = hh_utils.h_to_mesh_points(h)
            mesh = fd.UnitSquareMesh(mesh_points,mesh_points)
            V = fd.FunctionSpace(mesh, "CG", 1)

            f = 0.0
            d = fd.as_vector([1.0/fd.sqrt(2.0),1.0/fd.sqrt(2.0)])
            x = fd.SpatialCoordinate(mesh)
            nu = fd.FacetNormal(mesh)
            g=1j*k*fd.exp(1j*k*fd.dot(x,d))*(fd.dot(d,nu)-1)

            for noise_master in noise_master_level_list:
                A_noise_master = noise_master[0]
                n_noise_master = noise_master[1]

                for modifier in noise_modifier_list:

                    print(k,h_tuple,noise_master,modifier)
                    
                    A_modifier = h ** modifier[0] * k**modifier[1]
                    n_modifier = h ** modifier[2] * k**modifier[3]
                    A_noise_level = A_noise_master * A_modifier
                    n_noise_level = n_noise_master * n_modifier
                    A_stoch = coeff.PiecewiseConstantCoeffGenerator(
                        mesh,num_pieces,A_noise_level,A_pre,[2,2])
                    n_stoch = coeff.PiecewiseConstantCoeffGenerator(
                        mesh,num_pieces,n_noise_level,n_pre,[1])
                    np.random.seed(seed)
                    
                    GMRES_its = nearby_preconditioning_experiment(
                        V,k,A_pre,A_stoch,n_pre,n_stoch,f,g,num_repeats)

                    hh_utils.write_GMRES_its(
                        GMRES_its,save_location,
                        {'k' : k,
                         'h_tuple' : h_tuple,
                         'num_pieces' : num_pieces,
                         'A_pre_type' : A_pre_type,
                         'n_pre_type' : n_pre_type,
                         'noise_master' : noise_master,
                         'modifier' : modifier,
                         'num_repeats' : num_repeats
                         }
                        )

def nearby_preconditioning_experiment_gamma(k_range,n_lower_bound,n_var_base,
                                      n_var_k_power_range,num_repeats):
    """Tests the effectiveness of nearby preconditioning for a
    homogeneous but gamma-distributed random refractive index.

    This is an initial version - it holds the mean of the refractive
    index constant = 1, but then changes the variance of the refractive
    index.
    """
    
    for k in k_range:

        num_points = hh_utils.h_to_mesh_points(k**(-1.5))
        
        mesh = fd.UnitSquareMesh(num_points,num_points)

        V = fd.FunctionSpace(mesh, "CG", 1)
        
        for n_var_k_power in n_var_k_power_range:
            print(k)
            print(n_var_k_power)
            n_var = n_var_base * k**n_var_k_power
            
            # Ensure Gamma variates have mean 1 - n_lower_bound and
            # variance n_var
            scale = n_var / (1.0 - n_lower_bound)
            shape = (1.0 - n_lower_bound)**2 / n_var
            
            n_stoch = coeff.GammaConstantCoeffGenerator(
                shape,scale,n_lower_bound)

            n_pre = 1.0
            f = 0.0
            g = 1.0
            
            GMRES_its = nearby_preconditioning_test(
                V,k,A_pre=fd.as_matrix([[1.0,0.0],[0.0,1.0]]),
                A_stoch=None,n_pre=n_pre,n_stoch=n_stoch,
                f=f,g=g,num_repeats=num_repeats)

            save_location =\
                "/home/owen/Documents/code/helmholtz-firedrake/output/testing/"

            info = {"function" : "nearby_preconditioning_test_gamma",
                    "k" : k,
                    "h" : "k**(-1.5)",
                    "n_var_base" : n_var_base,
                    "n_var_k_power" : n_var_k_power,
                    "n_lower_bound" : n_lower_bound,
                    "scale" : scale,
                    "shape" : shape,
                    "f" : f,
                    "g" : g,
                    "n_pre" : n_pre,
                    "num_repeats" : num_repeats
                    }
                    
            
            hh_utils.write_GMRES_its(GMRES_its,save_location,info)

def test_fem_approx_props(k_list,h_mult_power_list,num_pieces,
                                     noise_level_system_A,noise_level_system_n,
                                     noise_level_rhs_A,num_system,num_rhs,
                                     fine_grid_mult_power,seed):
    """Tests to see if the required condition in the paper holds.

    For a variety of different Helmholtz systems, and a variety of
    different special right-hand sides (given by F(v) = (A_rhs
    \grad(\sum_i \alpha_i \phi_i),\grad v)_{L^2}, where the phi_i are
    the basis functions for piecewise-affine finite-elements, records
    the weighted H^1 norm of the finite-element error (approximated by
    taking the solution on a much finer grid) and the norm of the
    right-hand side in (H^1_k)'.

    Parameters:

    k_list - list of positive integers - the values of k for which
    experiments will be done.

    h_mult__power_list - list of 2-tuples, where each tuple consists of
    a positive real and a real. These define the magnitude of the
    different values of h For example, if h_mult_power_list =
    [(1.0,-1.0),(2.0,-1.5)] then two sets of experiments will be done,
    one with h = 1.0 * k**-1.0 and one with h = 2.0 * k**-1.5. Note that
    if the second element of any tuple is 0.0, then the corresponding
    computations will be performed on a mesh that is independent of k.

    num_pieces - postive integer - the random coefficients will be
    piecewise-constant on a num_pieces by num_pieces grid. (See notes in
    coefficients.PiecewiseConstantCoeffGenerator about the limits on
    num_pieces.)

    noise_level_system_A - positive real - the size of the random
    perturbations in the coefficient A defining the Helmholtz problem.

    noise_level_system_n - positive real - the size of the random
    perturbations in the coefficient n defining the Helmholtz problem.

    noise_level_rhs_A - positive real - the size of the random
    perturbations in the coefficient A defining right-hand side.

    num_system - positive integer - the number of different system
    matrices (i.e. A,n) for which to perform experiments.

    num_rhs - positive integer - the number of different right-hand
    sides (for each different system) for which to perform experiments.

    fine_grid_mult_power - 2-tuple of a positive real and a real - the
    mesh size on which the fine solution (which will serve as a proxy
    for the true solution) will be computed. E.g. if
    fine_grid_mult_power = (0.1,-2.0) then for each value of k the fine
    grid mesh size will be h = 0.1 * k**-2.0. Note that if the second
    element of the tuple is 0.0, then the mesh size will be independent
    of k.

    seed - positive integer, prime, not too large. Used to set the
    random seeds in the generation of the random coefficients.


    Output - For each k, the results are outputted to a Pandas DataFrame, the location of which is contained in 'k-df_functions_loc.json'.
    """

    # Test that both the h arrays are the same length?

    # For computational ease, add the fine mesh to the end of the list
    h_mult_power_list.append(fine_grid_mult_power)
    
    for k in k_list:            

        # Calculate number of points for all meshes
        ideal_mesh_sizes = [h_mult_power_list[ii][0] * k**h_mult_power_list[ii][1] for ii in range(num_h)]

        all_num_points = [utils.h_to_mesh_points(h) for h in ideal_mesh_sizes]

        # Set up storage
        index_labels = [h_mult_power_list,fine_grid_mult_power]

        column_labels = pd.MultiIndex.from_tensor([list(range(num_system)),list(range(num_rhs))])

        storage = pd.DataFrame(np.empty(num_h,num_system * num_rhs),columns=column_labels,index=index_labels)

        # Don't think this will work, as I want to be able to have each entry as a numpy array (the data)

        # The following will also calculate the solution on the fine mesh, because we added that to the end of h_mult_power_list
        for ii_h in range(len(h_mult_power_list)):
                        
            (prob,A_rhs,f_rhs) =\
                rhs_setup_for_fem_testing(all_num_points[ii_h],num_pieces,
                                        noise_level_system_A,
                                        noise_level_system_n,noise_level_rhs_A)

                for ii_system in range(num_system):

                # What follows with constantly setting seeds is a bit of
                # a hack - we need to get identical random numbers for
                # each different value of h, and this is the simplest
                # way to do it (that I can think of).

                # As random seeds, for the system use multiples of 2,
                # and for the right-hand sides use the odd multiples of
                # 3 (plus the input argument seed in both cases). Then
                # no seed is ever used twice.
                
                np.random.seed(seed + 2.0*ii_system)

                prob.A_stoch.sample()

                prob.n_stoch.sample()
                
                for ii_rhs in range(num_rhs):

                    print("k, h number, system number, rhs number")
                    print(k, ii_h, ii_system, ii_rhs)

                    np.random.seed(seed + 3.0 + 6.0 * ii_rhs)

                    A_rhs.sample()

                    # Assign random normal(0,1**2) to each entry of f. I
                    # am unsure if this will work, as you might need to
                    # assign an expression. However, if it doesn't
                    # we'll just hack it using the dat.
                    f_rhs.assign(np.random.normal(
                        f_rhs_coarse.vector().array().size))
                    
                    np.random.seed(seed + 3.0 + 6.0 * ii_rhs)

                    prob.solve()

                    # Save the function data
                    # No ide if this way of indexing will work.
                    storage[ii_h][ii_system][ii_rhs] = prob.u_h.dat.data_ro

# Need to make mesh_gen file - k-dependent aaaaah!

        file_loc = error.complex_write_functions(storage)

        # Rename file location so subsequent runs don't overwrite it
        move(file_loc,str(k) + file_loc)

# Need to figure out how to attach metadata - Sumatra? Don't worry for now.

def real_process_for_fem_approx_props(k_list):
    """Process the files generated in test_fem_approx_props.

    Currently, that means put them all on one graph.

    ONLY RUN THIS IN REAL FIREDRAKE.

    Inputs:

    k_list - see test_fem_approx_props. Must be the same k_list as used in the corresponding call to test_fem_approx_props.
    """

    # Select colormap tuples based on how many items there are in h_list - if we've got <= 10 mesh types, then tab10 is what we want, otherwise, tab20 (I think)
    colormap = cm.get_cmap('tab10')

    df_massive = 
    
    # For each k
    for k in k_list:
        df_out = error.real_process_functions(str(k) + 'df_functions_loc.json',norm_type=????)
        # I'm not entirely clear how to do the norm thing - I want to do it like a function handle, but that doesn't seem quite right here....
        

        # Plot results in a different colour for different mesh dependencies
        h_num = len(df_out.index)
        for ii_h in range(h_num):
            plt.plot(k,df_out,iloc[ii_h,:],color=colormap(floor(ii_h * 255 / h_num)))
    # Dsiplay the plot
    plt.show()
    # I full expect the legend will look an absolute mess.
    # Maybe it'll be better to bring it into a big dataframe, and then do some plotting based on multiindices.
    # Yes, let's do that instead.

        
                    
          
def rhs_setup_for_fem_testing(num_points,num_pieces,noise_level_system_A,
                            noise_level_system_n,noise_level_rhs_A):
    """Sets up all the problems for the experiments with a special rhs.

    Also ensures an exact direct solver (using an LU factorisation) is
    used.
    
    Parameters:

    num_points - positive integer - the mesh for the problem will be a
    num_points by num_points grid on the unit square.

    num_pieces - see  special_rhs_for_paper_experiment.

    noise_level_system_A - see  special_rhs_for_paper_experiment.

    noise_level_system_n - see  special_rhs_for_paper_experiment.

    noise_level_rhs_A  - see  special_rhs_for_paper_experiment.

    

    Outputs:

    Tuple (prob,A_rhs,f_rhs), where
    
    A_rhs - a matrix-valued realisation of
    PiecewiseConstantCoeffGenerator, defined by the parameters
    num_pieces and noise_level_rhs_A.

    f_rhs - a firedrake Function, initialised as all zeros.

    prob - a StochasticHelmholtzProblem with random coefficients defined
    by the parameters num_points, noise_level_system_A, and
    noise_level_system_n, and with a special right-hand side given by
    A_rhs and f_rhs.

    """

    mesh = fd.UnitSquareMesh(numpoints,num_points)

    A_system = coeff.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                     noise_level_system_A,
                                                     as_matrix(
                                                         [[1.0,0.0],
                                                          [0.0,1.0]]),
                                                     [2,2])

    n = coeff.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                              noise_level_system_n,1.0,[1])

    V = fd.FunctionSpace(mesh,"CG",1)
            
    prob = hh.StochasticHelmholtzProblem(k,V,A_stoch=A,n_stoch=n)

    A_rhs = coeff.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                  noise_level_rhs_A,
                                                  as_matrix([[1.0,0.0],
                                                             [0.0,1.0]]),
                                                  [2,2])

    f_rhs = fd.Function(mesh)

    prob.set_rhs_nbpc_paper(A_rhs.coeff,f_rhs)

    prob.force_lu()

    return (prob,A_rhs,f_rhs)
