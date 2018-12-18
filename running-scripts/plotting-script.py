import helmholtz_firedrake.utils as utils
from os import listdir
from fnmatch import fnmatch
import numpy as np
from matplotlib import pyplot as plt
this_directory = './'

csv_list = []
for filename in listdir():
    print(filename)
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

# This only comes into play when doing a lot of runs
#max_len = 0
#for ii in all_csvs_df.index:
#    this_output = np.unique(all_csvs_df.loc[ii,:].values)
#    max_len = max(max_len,len(this_output))
#    all_csvs_df.loc[ii,:(len(this_output)-1)] = this_output
#    all_csvs_df.loc[ii,len(this_output):] = np.nan

# Cut the nans

#all_csvs_dropped_df = all_csvs_df.drop(labels=[ii for ii in range(max_len,10000)],axis=1)



def plt_gmres(n_pre_type,noise_master,ks,modifiers):
    cols = ['k','r','b','g']
    markers = ['o','v','^']
    handles = []
    for ii in range(len(modifiers)):
        modifier = modifiers[ii]
        for k in ks:
            data = all_csvs_df.xs((n_pre_type,noise_master,modifier,k),level=('n_pre_type','noise_master','modifier','k'),drop_level=False)
            for jj in data.columns:
                if k == ks[0] and jj == 0:
                    handles.append(plt.scatter(x=data.reset_index().loc[0,'k'],y=data.iloc[0,jj],c=cols[ii],marker=markers[ii],label='Modifier = '+modifier))
                else:
                    plt.scatter(x=data.reset_index().loc[0,'k'],y=data.iloc[0,jj],c=cols[ii],marker=markers[ii],label='Modifier = '+modifier)
    #labels = # noise = noise_master non-zero $k^whatever$ # for now I'll just do the labels by hand.
    plt.legend(handles=handles)
    #title_string = r'$n^{(1)} = $' + n_pre_type# + ' and noise_master = ' + noise_master
    # Do title by hand
    title_string = 'Insert title here'
    plt.title(title_string)
    plt.show()

#----- Should only need to edit below here ------


n_pre_type = 'constant'

#noise_master = '(0.1, 0.0)' # To use with A
noise_master = '(0.0, 0.3)' # To use with n

ks = [10.0,20.0,30.0,40.0]

#modifiers = ['(0.0, -1.0, 0.0, 0.0)','(0.0, -0.5, 0.0, 0.0)','(0.0, 0.0, 0.0, 0.0)'] # to use with A
modifiers = ['(0.0, 0.0, 0.0, -1.0)','(0.0, 0.0, 0.0, -0.5)','(0.0, 0.0, 0.0, 0.0)'] # to use with n

# ------ An example -------

for ii in range(len(modifiers)):
    plt_gmres('constant',noise_master,ks,modifiers[:(ii+1)])

                              



