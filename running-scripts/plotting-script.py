import helmholtz_firedrake.utils as utils
from os import listdir
from fnmatch import fnmatch
import numpy as np
from matplotlib import pyplot as plt
this_directory = './'

csv_list = []
for filename in listdir():
    print(filename)
    if fnmatch(filename,'*CSV'):
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


max_len = 0
for ii in all_csvs_df.index:
    this_output = np.unique(all_csvs_df.loc[ii,:].values)
    max_len = max(max_len,len(this_output))
    all_csvs_df.loc[ii,:(len(this_output)-1)] = this_output
    all_csvs_df.loc[ii,len(this_output):] = np.nan

# Cut the nans
#for ii in range(max_len,10000):

all_csvs_dropped_df = all_csvs_df.drop(labels=[ii for ii in range(max_len,10000)],axis=1)

#def name2levels(name):
#    return all_csvs_dropped_df.index.levels[all_csvs_df.index.names.index(name)]

#for n_pre_type in name2levels('n_pre_type'):
#    for noise_master in name2levels('noise_master'):
        # Start plot
#        for modifier in name2levels('modifier'):
#            for k in name2levels('k'):
#                data = all_csvs_df.xs((n_pre_type,noise_master,modifier,k),level=('n_pre_type','noise_master','modifier','k'),drop_level=False)
                # idea from http://stackoverflow.com/questions/20084487/ddg#20084590
#                data.reset_index().plot()

#data = all_csvs_dropped_df.xs((n_pre_type,noise_master,modifier,k),level=('n_pre_type','noise_master','modifier','k'),drop_level=False)

# from https://stackoverflow.com/questions/28203643/plotting-multiple-lines-in-ipython-pandas-produces-multiple-plots

#ax = plt.axes()

n_pre_type = 'constant'

#noise_master = '(0.0, 0.1)'
noise_master = '(0.1, 0.0)'

#p = 1
def plt_gmres(n_pre_type,noise_master):
    ks = [10.0,20.0,30.0]
    cols = ['k','r','b','g']
    markers = ['o','v','^']
    #modifiers = ['(0.0, 0.0, 1.0, -1.0)','(0.0, 0.0, 1.0, -0.5)','(0.0, 0.0, 1.0, 0.0)']
    modifiers = ['(1.0, -1.0, 0.0, 0.0)','(1.0, -0.5, 0.0, 0.0)','(1.0, 0.0, 0.0, 0.0)']
    handles = []
    for ii in range(len(modifiers)):
        modifier = modifiers[ii]
        for k in ks:
            data = all_csvs_dropped_df.xs((n_pre_type,noise_master,modifier,k),level=('n_pre_type','noise_master','modifier','k'),drop_level=False)
            for jj in data.columns:
                if not np.isnan(data.iloc[0,jj]):
                    #print(data.iloc[0,jj],flush=True)
                    #print(data.reset_index().loc[0,'k'])
                    #data.reset_index().plot.scatter(x='k',y=jj,c=cols[ii],ax=ax)
                    if k == ks[0] and jj ==0:
                        handles.append(plt.scatter(x=data.reset_index().loc[0,'k'],y=data.iloc[0,jj],c=cols[ii],marker=markers[ii],label='Modifier = '+modifier))
                    else:
                        plt.scatter(x=data.reset_index().loc[0,'k'],y=data.iloc[0,jj],c=cols[ii],marker=markers[ii],label='Modifier = '+modifier)    
                    #print(data.iloc[0,jj] == np.nan)
    #                print(data.iloc[0,jj])
    #        data.reset_index().plot.scatter(x='k',y=data.columns,c=cols[ii],ax=ax)
    plt.legend(handles=handles,loc=5)
    # legend placement - ok for now
    # title
    title_string = 'n_pre_type = ' + n_pre_type + ' and noise_master = ' + noise_master
    plt.title(title_string)
    plt.show()


# Idea inspired by https://scentellegher.github.io/programming/2017/07/15/pandas-groupby-multiple-columns-plot.html


                              



