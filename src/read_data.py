import numpy as np
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

def extract_signals(path) -> Optional[Tuple]:

    def select():
        keys = select_keys(path, file)
        if len(keys)==2:
            for k in keys:
                if 'DE' in k:
                    d_end = file[k]
                elif 'FE' in k:
                    f_end = file[k]
            return np.hstack([d_end, f_end]) 
        else:
            return None

    file = loadmat(path)
    data = select()
    return data

def select_keys(path, dic):
    f_name = path.split('\\')[-1].split('_')[0]

    if f_name == '174':
        f_name = '173'
    
    if '@' in f_name:
        f_name = f_name.split('@')[0]

    i_key = [(f_name in key) and ('RPM' not in key) for key in dic.keys()]
    keys = [k for i, k in enumerate(dic.keys()) if i_key[i]]

    return keys

def new_path(path):
    new_path_f = path.split('.')[:-1]
    new_path_f.append('csv')
    new_path_f = '.'.join(new_path_f)
    return new_path_f
    
def save_csv():
    column_names = ['DE','FE']
    with open(new_path_f, 'w') as f:
        f.write(','.join(column_names) + '\n')
        np.savetxt(f, data, delimiter=',',fmt='%.10f')

data_dir = r'D:\Masters\CWRU-dataset\48k_Drive_End_Bearing_Fault_Data'

for root, dirs, files in os.walk(data_dir):
    if not (len(files) == 0) :
        path_files = [os.path.join(root,f) for f in files if f.endswith('.mat')]
        for path_f in path_files:
            data = extract_signals(path_f)
            new_path_f = new_path(path_f)
            save_csv()


