import h5py
import pandas as pd
import numpy as np

#This module read results from the CGNS files and returns a pandas dataframe.
#Save option decides if data is stored or not

def readerRes(file, save_option):
    resDir = 'C:\\Users\Lorenzo\Desktop\DATA\PROCESSED\\'
    print('Reading:', file)
    resName = file.replace('C:\\Users\Lorenzo\Desktop\DATA\LES_1.09m3s_export_', '')
    resName = resName.replace('e+01.cgns', '')
    f = h5py.File(file, 'r')
    region = f['Base']['rotor_new']
    sol = region['Solution0000001']  # access the solution
    fields = list(sol.keys())  # saves the stored field names as a list
    fields = fields[2::]  # drops trash
    df = pd.DataFrame(data=None, columns=fields)  # create an empty df
    for cname in fields:
        df[cname] = np.array(sol[cname][' data'][::1])
    df.columns = df.columns.str.replace("Monitor", "")
    if save_option == 'True':
        df.to_csv(resDir+'/NORM_DATA/'+resName+'.csv',index=False)
    return(df, resName)