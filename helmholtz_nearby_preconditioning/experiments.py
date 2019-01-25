# This is a hack to get PETSc to give me the timings like I want them.
import sys
print(sys.argv)
sys.argv.append('-log_view')
sys.argv.append('-history')
sys.argv.append('tmp.txt')
print(sys.argv)

import firedrake as fd
import helmholtz_firedrake.problems as hh
import helmholtz_firedrake.coefficients as coeff
import helmholtz_firedrake.utils as hh_utils
import numpy as np
import latticeseq_b2
from copy import deepcopy
import pandas as pd

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
        if fd.COMM_WORLD.rank == 0:
            print(ii_repeat)
        try:
            prob.solve()
        except RecursionError:
            print("Suffered a Python RecursionError.\
            Have you specified something using a big loop in UFL?\
            Aborting all further solves.")
            break

        if fd.COMM_WORLD.rank == 0:
        
            all_GMRES_its.append(prob.GMRES_its)

            prob.sample()

    all_GMRES_its  = np.array(all_GMRES_its)

    return all_GMRES_its

def nearby_preconditioning_piecewise_experiment_set(
        A_pre_type,n_pre_type,dim,num_pieces,seed,num_repeats,
        k_list,h_list,p_list,noise_master_level_list,noise_modifier_list,
        save_location):
    """Test nearby preconditioning for a range of parameter values.

    Performs nearby preconditioning tests for a range of values of k,
    the mesh size h, and the size of the random noise (which can be
    specified in terms of k and h). The random noise is piecewise
    constant on a grid unrelated to the finite-element mesh.

    Parameters:

    A_pre_type - string - options are 'constant', giving A_pre =
    [[1.0,0.0],[0.0,1.0]].

    n_pre_type - string - options are 'constant', giving n_pre = 1.0;
    'jump_down' giving n_pre = 2/3 on a central square of side length
    1/3, and 1 otherwise; and 'jump_up' giving n_pre = 1.5 on a central
    square of side length 1/3, and 1 otherwise.

    dim - 2 or 3, the dimension of the problem.

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

    p_list - list of positive ints, the polynomial degrees to run
    experiments for. Degree >= 5 will be very slow because of the
    implementation in Firedrake.

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


    
    for k in k_list:
        for h_tuple in h_list:
            for p in p_list:
                h = h_tuple[0] * k**h_tuple[1]
                mesh_points = hh_utils.h_to_num_cells(h,dim)
                mesh = fd.UnitSquareMesh(mesh_points,mesh_points)
                V = fd.FunctionSpace(mesh, "CG", p)
                f = 0.0
                d = fd.as_vector([1.0/fd.sqrt(2.0),1.0/fd.sqrt(2.0)])
                x = fd.SpatialCoordinate(mesh)
                nu = fd.FacetNormal(mesh)
                g=1j*k*fd.exp(1j*k*fd.dot(x,d))*(fd.dot(d,nu)-1)

                if A_pre_type is "constant":
                    A_pre = fd.as_matrix([[1.0,0.0],[0.0,1.0]])

                if n_pre_type is "constant":
                    n_pre = 1.0

                elif n_pre_type is "jump_down":
                    n_pre = (2.0/3.0)\
                            + hh_utils.nd_indicator(
                                x,1.0/3.0,
                                np.array([[1.0/3.0,2.0/3.0],
                                          [1.0/3.0,2.0/3.0],
                                          [1.0/3.0,2.0/3.0]]
                                )
                            )

                elif n_pre_type is "jump_up":
                    n_pre = 1.5\
                            + hh_utils.nd_indicator(
                                x,-1.0/2.0,
                                np.array([[1.0/3.0,2.0/3.0],
                                          [1.0/3.0,2.0/3.0],
                                          [1.0/3.0,2.0/3.0]]
                                )
                            )


                for noise_master in noise_master_level_list:
                    A_noise_master = noise_master[0]
                    n_noise_master = noise_master[1]

                    for modifier in noise_modifier_list:
                        if fd.COMM_WORLD.rank == 0:
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

def qmc_nbpc_experiment(h_spec,dim,J,M,k,delta,lambda_mult,
                        mean_type,use_nbpc,GMRES_threshold):
    """Performs QMC for the Helmholtz Eqn with nearby preconditioning.

    Mention: expansion, n only, unit square, the idea of the algorithm.

    Parameters:

    h_spec - like one entry of h_list in piecewise_experiment_set.

    dim - 2 or 3 - the spatial dimension.

    J - positive int - the length of the KL-like expansion in the
    definition of n.

    M - positive int - 2**M is the number of QMC points to use.

    k - positive float - the wavenumber.

    delta - see the definition of delta in
    helmholtz_firedrake.coefficients.UniformKLLikeCoeff.__init__.

    lambda_mult - see the definition of lambda_mult in
    helmholtz_firedrake.coefficients.UniformKLLikeCoeff.__init__.

    mean_type - one of 'constant', INSERT MORE IN HERE - n_0 in the
    expansion for n.

    use_nbpc - Boolean - whether to use nearby preconditioning to speed
    up the qmc method, or to perform an LU decomposition for each QMC
    point, and use this to precondition GMRES (which will converge in
    one step).

    point_generation_method - either 'qmc' or 'mc'. 'qmc' means a QMC
    lattice rule is used to generate the points, whereas 'mc' means the
    points are randomly generated according to a uniform distribution on
    the cube.

    seed - seed with which to start the randomness.

    GMRES_threshold - positive int - the number of GMRES iteration we
    will tolerate before we 'redo' the preconditioning.

    Outputs:

    time - a 2-tuple of non-negative floats. time[0] is the amount of
    time taken to calculate the LU decompositions, and time[1] is the
    amount of time taken performing GMRES solves. NOT YET IMPLEMTENTED.

    points_info - a pandas DataFrame of length M, where each row
    corresponds to a QMC point. The columns of this dataframe are 'sto_loc'
    - a numpy array giving the location of the point in stochastic
    space; 'LU' - a boolean stating whether the system matric
    corresponding to that point was factorised into its LU
    factorisation; and 'GMRES' - a non-negative int giving the number of
    GMRES iterations it took to achieve convergence at this QMC
    point. The points are ordered (from top to bottom) in the order they
    were tackled by the algorithm.

    """
    scaling = lambda_mult * np.array(list(range(1,J+1)),
                                     dtype=float)**(-1.0-delta)

    if point_generation_method is 'qmc':
        # Generate QMC points on [-1/2,1/2]^J using Dirk Nuyens' code
        qmc_generator = latticeseq_b2.latticeseq_b2(s=J)

        points = []

        # The following range will have M as its last term
        for m in range((M+1)):
            points.append(qmc_generator.calc_block(m))

        qmc_points = points[0]

        for ii in range(1,len(points)):
            qmc_points = np.vstack((qmc_points,points[ii]))

        qmc_points -= 0.5

    elif points_generation_method is 'mc':
        qmc_points = np.random.rand(2**M,J)
    
    # Create the coefficient
    if mean_type is 'constant':
        n_0 = 1.0

    mesh_points = hh_utils.h_to_num_cells(h_spec[0]*k**h_spec[1],dim)
    mesh = fd.UnitSquareMesh(mesh_points,mesh_points)
    
    kl_like = coeff.UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,qmc_points)
    
    # Create the problem
    V = fd.FunctionSpace(mesh,"CG",1)
    # The Following lines are a hack because deepcopy isn't implemented
    # for ufl expressions (in general at least), and creating an
    # instance of the coefficient with particular 'stochastic
    # coordinates' is the easiest way to get the preconditioning
    # coefficient.

    prob = hh.StochasticHelmholtzProblem(k,V,None,kl_like,
                                         **{'A_pre' :
                                            fd.as_matrix([[1.0,0.0],[0.0,1.0]])
                                         })

    points_info_columns = ['sto_loc','LU','GMRES']
    points_info = pd.DataFrame(None,columns=points_info_columns)
    
    centre = np.zeros((1,J))

    # Order points relative to the origin
    order_points(prob.n_stoch.stochastic_points,centre,scaling)
    LU = np.nan
    prob.n_stoch.first_row_assign()

    update_pc(prob,mesh,J,delta,lambda_mult,n_0)
    
    num_solves = 0
    
    # The total number of points we have 'left' is
    # prob.n_stoch.stochastic_points.shape[0]
    while prob.n_stoch.stochastic_points.shape[0] > 0:

        prob.solve()
        num_solves += 1

        if use_nbpc is True:
            # If GMRES iterations were too big, or we're not using nbpc,
            # recalculate preconditioner.
            if (prob.GMRES_its > GMRES_threshold):
                
                new_centre(prob,mesh,J,delta,lambda_mult,n_0,scaling)
               

            else:               
                # Copy details of last solve into output dataframe
                temp_df = pd.DataFrame(
                    [[prob.n_stoch.current_point(),LU,prob.GMRES_its]],columns=points_info_columns)
                points_info = points_info.append(temp_df,ignore_index=True)

                try:
                    prob.sample()
                except coeff.SamplingError:
                    pass

        else: # Not using NBPC
            temp_df = pd.DataFrame(
                [[prob.n_stoch.current_point(),LU,prob.GMRES_its]],columns=points_info_columns)
            points_info = points_info.append(temp_df,ignore_index=True)

            try:
                prob.sample()
                new_centre(prob,mesh,J,delta,lambda_mult,n_0,scaling)
            except coeff.SamplingError:
                pass
            


        
    # Now trying to see if I can do stuff with timings
    # Try uing the re regular expression package
    
    
    try:
        open('tmp.txt','r')
        print('SUCCESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(num_solves)
    except:
        pass
    
    return points_info

def update_pc(prob,mesh,J,delta,lambda_mult,n_0):
    # Update the preconditioner
    n_pre_instance = coeff.UniformKLLikeCoeff(mesh,J,delta,lambda_mult,
                                              n_0,
                                              np.array(
                                                  deepcopy(prob.n_stoch.current_point()),
                                                  ndmin=2))
    
    prob.set_n_pre(n_pre_instance.coeff)

def new_centre(prob,mesh,J,delta,lambda_mult,n_0,scaling):
    centre = prob.n_stoch.current_point()
    order_points(prob.n_stoch.stochastic_points,centre,scaling)
    
    update_pc(prob,mesh,J,delta,lambda_mult,n_0)
    

def order_points(points,centre,scaling):
    """Orders the points in their distance to centre.

    Ordering is done with the dimensions scaled by the elements of
    scaling.

    Parameters:

    points - Numpy array with J columns, each row giving the location of
    a QMC point.

    centre - vector in $\mathbb{R}^J$

    scaling - numpy array of length J, giving the scaling in each
    dimension. Each element should be a positive float.

    Output

    Rearranges points IN PLACE to reflect the ordering.
    """
    
    distances = np.abs(points-centre) @ scaling

    # Sort the distances and extract the ordering

    ordering = np.argsort(distances)
    
    points_copy = deepcopy(points)

    for ii in range(points_copy.shape[0]):
        points[ii,:] = points_copy[ordering[ii],:]
