import csv
import pandas as pd
from os import listdir
from fnmatch import fnmatch

k_list = [10.0,20.0,30.0]

df_master = pd.DataFrame(index=[int(k) for k in k_list],columns=['num_lu','total_solves','lu_as_percentage','av_gmres','max_gmres'])

for k in k_list:
    
    this_directory = './'

    for filename_tmp in listdir():
        
        if fnmatch(filename_tmp,'k-'+str(k)+'*csv'):
            filename = filename_tmp

    df = pd.read_csv(filename,usecols=[2,3])

    df_master.loc[k,'num_lu'] = df.LU.sum()

    df_master.loc[k,'total_solves'] = df.LU.count()

    df_master.loc[k,'lu_as_percentage'] = 100.0*df.LU.mean()

    df_master.loc[k,'av_gmres'] = df.GMRES.mean()

    df_master.loc[k,'max_gmres'] = df.GMRES.max()

print(df_master)

column_names = ['Number of LU factorisations calculated','Total number of linear systems solved','Number of LU factorisations as percentage of total solves','Average number of GMRES iterations','Maximum number of GMRES iterations']

float_format = '{:.2f}'.format # Based on formatting described at https://pyformat.info/#number
# Helped debug using https://stackoverflow.com/a/20937592

column_format = 'Sc Sc Sc Sc Sc'

with open('table.tex',mode='w') as table:
    df_master.to_latex(table,header=column_names,float_format=float_format)

            
            

