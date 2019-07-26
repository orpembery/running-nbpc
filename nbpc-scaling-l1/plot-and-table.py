import pandas as pd
import helmholtz_firedrake.utils as utils
from os import listdir
from fnmatch import fnmatch

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

print(df)
