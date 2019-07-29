import pandas as pd
import helmholtz_firedrake.utils as utils
from os import listdir
from fnmatch import fnmatch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

this_directory = './'

csv_list = []
for filename in listdir():
    if fnmatch(filename,'*csv'):
        csv_list.append(this_directory + filename)

num_eps = len(csv_list)

info_data_tmp = utils.read_repeats_from_csv(this_directory+csv_list[0])

num_k = len(info_data_tmp[1])

k_list = [ii[1] for ii in info_data_tmp[1]]

data = []

eps_list = []
        
for file in csv_list:
        
    info_data = utils.read_repeats_from_csv(this_directory+file)

    string = file

    # A bit of a hack, because I know the filenames
    eps = float(string[19:22])

    eps_list.append(eps)
    
    this_k_data = [ii[2] for ii in info_data[1]]

    data.append(this_k_data)
    
df = pd.DataFrame(data,columns=k_list,index=eps_list)

df = df.sort_index()

# Put dataframe in a table

# Plot results for 0.0 to 0.3 on one axis 0.4-0.7 on another and 0.8-1.0 on another

def make_plot(locs):

    fig = plt.figure()

    styles = 'ovsd'

    for ii_loc in range(len(locs)):

        loc = locs[ii_loc]

        # special for 0.0 and 1.0


        if loc == 1.0:
            str_loc = ''
        else:
            str_loc = str(loc)


        if loc == 0.0:
            label = r'$\alpha = 0.2}$'
        else:
            label = r'$\alpha = 0.2/k^{'+str_loc+'}$'

        
        
        df.loc[loc,:].T.plot(style='k'+styles[ii_loc]+'--',label=label)

    plt.xlabel('$k$')

    plt.ylabel('Number of GMRES iterations')

    plt.xlim([0,110])

    plt.xticks([10,20,30,40,50,60,70,80,90,100])

    # Found out about this from https://www.scivision.dev/matplotlib-force-integer-labeling-of-axis/
    ax = fig.gca()
    
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
    plt.legend(loc='upper left')
    
    plt.show()


# Make the plots
    
make_plot([0.0,0.1,0.2,0.3])

make_plot([0.4,0.5,0.6,0.7])

make_plot([0.8,0.9,1.0])


