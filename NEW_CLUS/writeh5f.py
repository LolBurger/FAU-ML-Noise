import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import glob
from sklearn.preprocessing import RobustScaler
import pickle
import os


slicename = ['-0.012P','0P','0.01P','0.021','hub','mid','tip']
fileDir = r'C:\Users\lorenzo\Desktop\\'
hffile = fileDir + 'totalData.h5'

#change slicename index to get the correct slices
slice = slicename[5]


flist = glob.glob(fileDir+slice+'\*.csv')
header = ['dUrdr','dUrdt','dUrdx','dUtdr','dUtdt','dUtdx','dUxdr','dUxdt','dUxdx',
          'SP', 'Ur', 'Ut', 'Ux', 'WD', 'dPdr', 'dPdt', 'dPdx',
          'TKE_2', 'PCWE', 'PCWE_1', 'X', 'Y', 'Z'
          ]

header2 = ['dUrdr', 'dUrdt', 'dUrdx', 'dUtdr', 'dUtdt', 'dUtdx', 'dUxdr', 'dUxdt',
       'dUxdx', 'SP', 'Ur', 'Ut', 'Ux', 'WD', 'dPdr', 'dPdt', 'dPdx', 'PCWE',
        'TKE']


flist = flist[2:]



res = pd.DataFrame(data=None,columns=header2)
for aa, file in enumerate(flist):
    datum = pd.read_csv(file,names=header,skiprows=1)
    #datum['PCWE']=datum['PCWE'].astype('float')
    #datum.replace([np.inf, -np.inf], np.nan, inplace=True)
    #datum = datum.dropna()
    datum['TKE']=0.5*(datum['Ur']**2+datum['Ut']**2+datum['Ux']**2)
    datum['Theta']  = np.arctan(datum['Y']/datum['Z'])
    datum = datum.drop(columns=['TKE_2','PCWE_1','X','Y','Z','Theta'])
    res = pd.concat([res,datum])

transformer = RobustScaler().fit(res)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(res.drop(columns=['PCWE']))

print(np.cumsum(pca.explained_variance_ratio_))
hffilePCA = fileDir + 'totalDataPCA.h5'
hf = h5py.File(hffilePCA, 'w')
gruppo = hf.create_group(slice)
for aa, file in enumerate(flist):
    datum = pd.read_csv(file,names=header,skiprows=1)
    #datum['PCWE']=datum['PCWE'].astype('float')
    #datum.replace([np.inf, -np.inf], np.nan, inplace=True)
    #datum = datum.dropna()
    datum['TKE']=0.5*(datum['Ur']**2+datum['Ut']**2+datum['Ux']**2)
    datum['Theta']  = np.arctan(datum['Y']/datum['Z'])
    coords = datum[['X','Theta']]
    datum = datum.drop(columns=['TKE_2','PCWE_1','X','Y','Z','Theta'])
    gruppo2 = gruppo.create_group('slice_'+str(aa))
    dati = transformer.transform(datum)
    sel2 = pd.DataFrame(dati,columns=datum.columns)
    sel = sel2['PCWE']

    dati = pca.transform(sel2.drop(columns=['PCWE']))
    dati = dati[:,:3]
    dati = pd.concat([pd.DataFrame(dati,columns=['Score_1','Score_2','Score_3']),sel,coords], axis=1)
    for jj in dati.columns:
        gruppo2[jj] = dati[jj]

hf.close()

hf = h5py.File(hffile, 'w')
gruppo = hf.create_group(slice)
for aa, file in enumerate(flist):
    datum = pd.read_csv(file,names=header,skiprows=1)
    #datum['PCWE']=datum['PCWE'].astype('float')
    #datum.replace([np.inf, -np.inf], np.nan, inplace=True)
    #datum = datum.dropna()
    datum['TKE']=0.5*(datum['Ur']**2+datum['Ut']**2+datum['Ux']**2)
    datum['Theta']  = np.arctan(datum['Y']/datum['Z'])
    coords = datum[['X','Theta']]
    datum = datum.drop(columns=['TKE_2','PCWE_1','X','Y','Z','Theta'])
    gruppo2 = gruppo.create_group('slice_'+str(aa))
    dati = transformer.transform(datum)
    dati = pd.concat([pd.DataFrame(dati,columns=datum.columns),coords], axis=1)
    for jj in dati.columns:
        gruppo2[jj] = dati[jj]

hf.close()



print('end')