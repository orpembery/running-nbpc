import helmholtz_firedrake.utils as utils
from os import listdir
from fnmatch import fnmatch
import numpy as np
from matplotlib import pyplot as plt
import tikzplotlib

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

#print(all_csvs_df)

# This only comes into play when doing a lot of runs
#max_len = 0
#for ii in all_csvs_df.index:
#    this_output = np.unique(all_csvs_df.loc[ii,:].values)
#    max_len = max(max_len,len(this_output))
#    all_csvs_df.loc[ii,:(len(this_output)-1)] = this_output
#    all_csvs_df.loc[ii,len(this_output):] = np.nan

def plt_gmres(n_pre_type,noise_master,ks,modifiers):
    cols = ['k','r','b','g']
    markers = ['o','v','^']
    handles = []
    for ii in range(len(modifiers)):
        modifier = modifiers[ii]
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
                            c=cols[ii],
                            marker=markers[ii],
                            label='Modifier = '+modifier))
                else:
                    plt.scatter(
                        x=data.reset_index().loc[0,'k'],
                        y=data.iloc[0,jj],
                        c=cols[ii],
                        marker=markers[ii],
                        label='Modifier = '+modifier)
                    
    labels = [r'$\mathrm{noise\,\,level}= 0.1/k$',
              r'$\mathrm{noise\,\,level} = 0.1/(k^{0.5})$',
              r'$\mathrm{noise\,\,level} = 0.1$']
    labels = labels[:(len(modifiers)+1)]
    #plt.legend(handles,labels,loc=2)
    
    plt.xlabel(r'$k$')
    plt.ylabel('\# GMRES Iterations')

    plt.xticks([20,40,60,80]) # told by http://stackoverflow.com/questions/12608788/ddg#12608937

    # Yes, this is a hack. No, I don't care.
    plot_num = len(modifiers)
    if noise_master == '(0.1, 0.0)':
        if plot_num == 1:
            plt.yticks([7,8,9])
        elif plot_num == 2:
            plt.yticks([10,20])
        elif plot_num == 3:
            plt.yticks([40,80,120,160])
    elif noise_master == '(0.0, 0.1)':
        # if plot_num == :
        #     plt.yticks([4,5,6])
        # elif plot_num == 2:
        #     plt.yticks([5,10,15])
        # elif plot_num == 3:
        plt.yticks(#[25,50])
            #[100,200])
            #[250,500])
            #[500,1000])
            [4,5,6])
    #plt.show()

    plt.xlim([10,90])

    #plt.figure(num=1,figsize=(1,1))

    #https://stackoverflow.com/a/4306340
    
    fig = plt.gcf()
    
    fig.set_size_inches(3.5,3.5)
    
    plt.savefig('n-const-1.pgf')

    #tikzplotlib.save("n-const-1.tex")
    
    plt.show()
    
#----- Should only need to edit below here ------


n_pre_type = 'constant'

#noise_master = '(0.1, 0.0)' # To use with A
noise_master = '(0.0, 0.1)' # To use with n

#ks = [10.0,20.0,30.0,50.0,60.0,70.0,80.0,90.0,100.0]

ks = [20.0]#,40.0]#,60.0]#,80.0]

#ks = [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]

#modifiers = ['(0.0, -1.0, 0.0, 0.0)','(0.0, -0.5, 0.0, 0.0)','(0.0, 0.0, 0.0, 0.0)'] # to use with A
modifiers = ['(0.0, 0.0, 0.0, -1.0)','(0.0, 0.0, 0.0, -0.5)','(0.0, 0.0, 0.0, 0.0)'] # to use with n

# ------ An example -------

plt_gmres(n_pre_type,noise_master,ks,[modifiers[0]])

                              



