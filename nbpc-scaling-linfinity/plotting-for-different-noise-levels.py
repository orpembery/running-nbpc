import helmholtz_firedrake.utils as utils
from os import listdir
from fnmatch import fnmatch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

this_directory = './'

csv_list = []
for filename in listdir():
    if fnmatch(filename,'*csv'):
        csv_list.append(this_directory + filename)

info_data = utils.read_repeats_from_csv(this_directory+csv_list[0])

names_list = list(info_data[0].keys())

names_list.remove('num_pieces')
names_list.remove('h_tuple')
names_list.remove('Git hash')
names_list.remove('Date/Time')
names_list.remove('A_pre_type')
names_list.remove('num_repeats')
        
all_csvs_df = utils.csv_list_to_dataframe(csv_list,names_list)

def plt_gmres2(n_pre_type,noise_masters,ks,modifier):
    # Idea for old plotting script - loop over modifiers, loop over k, for each one, plot it
    # Idea for new script, loop over noise_masters, loop over k, for each one, plot it
    """Modifier must be a string (and noise_master)"""

    fig = plt.figure()
    
    handles = []
    for ii in range(len(noise_masters)):
        noise_master = noise_masters[ii]

        for k in ks:
            
            data = all_csvs_df.xs(
                (n_pre_type,noise_master,modifier,k),
                level=('n_pre_type','noise_master','modifier','k'),
                drop_level=False)
            
            for jj in data.columns:
                if k == ks[0] and jj == 0:
                    handles.append(
                        plt.scatter(
                            x=data.reset_index().loc[0,'k'],
                            y=data.iloc[0,jj],
                            c='k'))
                else:
                    plt.scatter(
                        x=data.reset_index().loc[0,'k'],
                        y=data.iloc[0,jj],
                        c='k')
    
    plt.xlabel(r'$k$')
    plt.ylabel('# GMRES Iterations')

    plt.xticks([20,40,60,80]) # told by http://stackoverflow.com/questions/12608788/ddg#12608937

    # Found out about this from https://www.scivision.dev/matplotlib-force-integer-labeling-of-axis/
    ax = fig.gca()
        
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

    plt.close(fig)

#----- Should only need to edit below here ------


n_pre_type = 'constant'

noise_master_A = '(0.5, 0.0)'

noise_master_n = '(0.0, 0.5)'

ks = [20.0,40.0,60.0,80.0]

A_modifiers = ['(0.0, 0.0, 0.0, 0.0)','(0.0, -0.5, 0.0, 0.0)','(0.0, -1.0, 0.0, 0.0)']

n_modifiers = ['(0.0, 0.0, 0.0, 0.0)','(0.0, 0.0, 0.0, -0.5)','(0.0, 0.0, 0.0, -1.0)']

# ------ An example -------
for modifier in A_modifiers:
    plt_gmres2(n_pre_type,[noise_master_A],ks,modifier)

for modifier in n_modifiers:
    plt_gmres2(n_pre_type,[noise_master_n],ks,modifier)
    
                              



