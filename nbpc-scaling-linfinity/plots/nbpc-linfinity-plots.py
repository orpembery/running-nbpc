import helmholtz_firedrake.utils as utils
from os import listdir
from fnmatch import fnmatch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import colorcet as cc
from matplotlib import rc, rcParams
import sys

# Have options depending on what plots to produce
# Options are:
# 1 - 100 repeats
# 2 - 200 repeats
# 3 - 10x10 deterministic, n only
# 4 - 2x2 deterministic, n only
plot_type = int(sys.argv[1])

div_thresh = 500

rc('text', usetex=True) # Found out about this from https://stackoverflow.com/q/54827147

rcParams.update({'text.latex.preamble':[r'\usepackage[urw-garamond]{mathdesign}',r'\usepackage[T1]{fontenc}'],'font.size':11})

if plot_type == 1:
    this_directory = '../output/'
elif plot_type == 2:
    this_directory = '../output-200-repeats/'
elif plot_type == 3:
    this_directory = '../output-deterministic-10/'
elif plot_type == 4:
    this_directory = '../output-deterministic-2/'
    
noise_level = 0.5

csv_list = []
for filename in listdir(this_directory):
    if fnmatch(filename,'*csv'):
        csv_list.append(this_directory+filename)

info_data = utils.read_repeats_from_csv(csv_list[0])

names_list = list(info_data[0].keys())

names_list.remove('num_pieces')
names_list.remove('h_tuple')
names_list.remove('Git hash')
names_list.remove('Date/Time')
names_list.remove('A_pre_type')
names_list.remove('num_repeats')
        
all_csvs_df = utils.csv_list_to_dataframe(csv_list,names_list)

def plt_gmres(n_pre_type,noise_master,ks,modifiers,filename,things_for_plotting):

    styles = 'o^v>P'

    colours = cc.glasbey_bw
    
    fig = plt.figure()

    use_nice_y_axis = False

    for modifier in modifiers:

        ii = modifiers.index(modifier)

        # Define y_data and x_data

        x_data = 'setup'

        y_data = 'setup'

        if things_for_plotting[ii] == 0.0:
            number = 0
        elif things_for_plotting[ii] == 1.0:
            number = 1
        else:
            number = things_for_plotting[ii]

        label = r'$\beta = $'+str(number)

        diverge_x = np.array([])
        
        for k in ks:
            data = all_csvs_df.xs((n_pre_type,noise_master,modifier,k),level=('n_pre_type','noise_master','modifier','k'),drop_level=False)

            data = data.to_numpy()

            # In case there's no data
            if data.size == 0:
                data = np.nan

            y_data_tmp = np.max(np.unique(data))

            #print(y_data_tmp)
            
            # In case GMRES diverged (i.e., GMRES itself diverged, or converged but took >= 500 iterations).
            if np.isinf(y_data_tmp) or y_data_tmp >= div_thresh:
                diverge_x = np.append(diverge_x,k)
            
            #print(y_data_tmp)

            if y_data_tmp < div_thresh:

                if np.all(y_data == 'setup'):
                    y_data = np.array(y_data_tmp)
                else:
                    y_data = np.append(y_data,y_data_tmp)

                if np.all(x_data == 'setup'):
                    x_data = np.array(k)
                else:
                    x_data = np.append(x_data,k)

            #print(x_data)
            #print(y_data)

        #print(x_data)

        #print(y_data)
                
        plt.plot(x_data,y_data,styles[ii]+'--',label=label,c=colours[ii])


        if diverge_x.size != 0:
            use_nice_y_axis = True
            #print('PLOTTING')
            #print(np.max(y_data))
            all_data = all_csvs_df.to_numpy()
            plt.plot(diverge_x,np.repeat(1.05*div_thresh,diverge_x.size),styles[ii],c='xkcd:gray')

        ax = fig.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
                     
    plt.xlabel(r'$k$')
    plt.ylabel(r'\textrm{Maximum Number of GMRES Iterations}')

    plt.legend()

    plt.xticks([float(k) for k in ks]) # told by http://stackoverflow.com/questions/12608788/ddg#12608937

    # Integers only on y axis
    # Found out about this from https://www.scivision.dev/matplotlib-force-integer-labeling-of-axis/
    ax = fig.gca()
    if use_nice_y_axis:
        plt.yticks([100,200,300,400,500])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # Maybe add an argument to MaxNLocator to give the number of intervals on the x axis
    fig.set_size_inches((5.5,5.5))
    
    plt.savefig(filename+'.pgf')
    plt.close('all')

#----- Should only need to edit below here ------


n_pre_type = 'constant'

noise_masters = ['('+str(noise_level)+', 0.0)','(0.0, '+str(noise_level)+')']

if plot_type == 3:
    ks = [20.0,40.0,60.0,80.0,100.0,120.0,140.0]
else:
    ks = [20.0,40.0,60.0,80.0,100.0]






# Need to sort saving names
# THis is a hack because my computational code saved things wrong
# in the if statements the first element of modifierss may not work, but haven't done A runs at this stage

if plot_type == 3:
    modifierss = [['(0.0, 0.0, 0.0, 0.0)', '(0.0, -0.1, 0.0, 0.0)', '(0.0, -0.2, 0.0, 0.0)', '(0.0, -0.3, 0.0, 0.0)', '(0.0, -0.4, 0.0, 0.0)', '(0.0, -0.5, 0.0, 0.0)', '(0.0, -0.6, 0.0, 0.0)', '(0.0, -0.7, 0.0, 0.0)', '(0.0, -0.8, 0.0, 0.0)', '(0.0, -0.9, 0.0, 0.0)', '(0.0, -1.0, 0.0, 0.0)'], ['(0.0, 0.0, 0.0, -0.0)', '(0.0, 0.0, 0.0, -0.1)', '(0.0, 0.0, 0.0, -0.2)', '(0.0, 0.0, 0.0, -0.3)', '(0.0, 0.0, 0.0, -0.4)', '(0.0, 0.0, 0.0, -0.5)', '(0.0, 0.0, 0.0, -0.6)', '(0.0, 0.0, 0.0, -0.7)', '(0.0, 0.0, 0.0, -0.8)', '(0.0, 0.0, 0.0, -0.9)', '(0.0, 0.0, 0.0, -1.0)']]
elif plot_type == 4:
    modifierss = [['(0.0, 0.0, 0.0, 0.0)', '(0.0, -0.1, 0.0, 0.0)', '(0.0, -0.2, 0.0, 0.0)', '(0.0, -0.3, 0.0, 0.0)', '(0.0, -0.4, 0.0, 0.0)', '(0.0, -0.5, 0.0, 0.0)', '(0.0, -0.6, 0.0, 0.0)', '(0.0, -0.7, 0.0, 0.0)', '(0.0, -0.8, 0.0, 0.0)', '(0.0, -0.9, 0.0, 0.0)', '(0.0, -1.0, 0.0, 0.0)'], ['(0.0, 0.0, 0.0, -0.0)', '(0.0, 0.0, 0.0, -0.1)', '(0.0, 0.0, 0.0, -0.2)', '(0.0, 0.0, 0.0, -0.3)', '(0.0, 0.0, 0.0, -0.4)', '(0.0, 0.0, 0.0, -0.5)', '(0.0, 0.0, 0.0, -0.6)', '(0.0, 0.0, 0.0, -0.7)', '(0.0, 0.0, 0.0, -0.8)', '(0.0, 0.0, 0.0, -0.9)', '(0.0, 0.0, 0.0, -1.0)']]
else:
    modifierss = [['(0.0, 0.0, 0.0, 0.0)', '(0.0, -0.1, 0.0, 0.0)', '(0.0, -0.2, 0.0, 0.0)', '(0.0, -0.3, 0.0, 0.0)', '(0.0, -0.4, 0.0, 0.0)', '(0.0, -0.5, 0.0, 0.0)', '(0.0, -0.6, 0.0, 0.0)', '(0.0, -0.7, 0.0, 0.0)', '(0.0, -0.8, 0.0, 0.0)', '(0.0, -0.9, 0.0, 0.0)', '(0.0, -1.0, 0.0, 0.0)'], ['(0.0, 0.0, 0.0, 0.0)', '(0.0, 0.0, 0.0, -0.1)', '(0.0, 0.0, 0.0, -0.2)', '(0.0, 0.0, 0.0, -0.3)', '(0.0, 0.0, 0.0, -0.4)', '(0.0, 0.0, 0.0, -0.5)', '(0.0, 0.0, 0.0, -0.6)', '(0.0, 0.0, 0.0, -0.7)', '(0.0, 0.0, 0.0, -0.8)', '(0.0, 0.0, 0.0, -0.9)', '(0.0, 0.0, 0.0, -1.0)']]

things_for_plotting = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

#modifierss = [['(0.0, -1.0, 0.0, 0.0)','(0.0, -0.5, 0.0, 0.0)','(0.0, 0.0, 0.0, 0.0)'],['(0.0, 0.0, 0.0, -1.0)','(0.0, 0.0, 0.0, -0.5)','(0.0, 0.0, 0.0, 0.0)']]

# ------ An example -------

# Fix me

plot_collection = [[0,4],[4,8],[8,11]]

for ii_An in range(2):
    #print('start-An-'+str(ii_An))
    
    if ii_An == 0 and (plot_type == 3 or plot_type == 4):
        continue # Only n plots for deterministic at this stage
    
    noise_master = noise_masters[ii_An]
    modifiers = modifierss[ii_An]

    if plot_type == 1:
        filename = 'nbpc-linfinity-plot-'
    elif plot_type == 2:
        filename = 'nbpc-linfinity-plot-200-repeats-'
    elif plot_type == 3:
        filename = 'nbpc-linfinity-plot-deterministic-10-'
    elif plot_type == 4:
        filename = 'nbpc-linfinity-plot-deterministic-2-'
    
    if ii_An == 0:
        filename += 'A'
    else:
        filename += 'n'
    filename += '-'
    
    for ii in range(len(plot_collection)):
        #print('start-'+str(ii))
        filename_tmp = filename + str(ii)
        
        plt_gmres(n_pre_type,noise_master,ks,modifiers[plot_collection[ii][0]:plot_collection[ii][1]],filename_tmp,things_for_plotting[plot_collection[ii][0]:plot_collection[ii][1]])
        #print('end-'+str(ii))
    #print('end-An-'+str(ii_An))
#print('end of file')

                              



