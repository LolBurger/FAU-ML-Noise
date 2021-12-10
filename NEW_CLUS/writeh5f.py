import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import glob
from sklearn.preprocessing import RobustScaler, StandardScaler
import pickle
import os


#slicename = ['-0.012P','0P','0.01P','0.021','hub','mid','tip']
slicename = ['hub','mid','tip']
fileDir = r'C:\Users\lorenzo\Desktop\\'
fileDir = r'C:\Users\Utente\Desktop\dataexp\\'
hffile = fileDir + 'totalData.h5'

#change slicename index to get the correct slices

header = ['dUrdr','dUrdt','dUrdx','dUtdr','dUtdt','dUtdx','dUxdr','dUxdt','dUxdx',
          'SP', 'Ur', 'Ut', 'Ux', 'WD', 'dPdr', 'dPdt', 'dPdx',
          'TKE_2', 'PCWE', 'PCWE_1', 'X', 'Y', 'Z'
          ]

header2 = ['dUrdr', 'dUrdt', 'dUrdx', 'dUtdr', 'dUtdt', 'dUtdx', 'dUxdr', 'dUxdt',
       'dUxdx', 'SP', 'Ur', 'Ut', 'Ux', 'WD', 'dPdr', 'dPdt', 'dPdx', 'PCWE',
        'TKE']

hffilePCA = fileDir + 'totalDataPCA.h5'
hf2 = h5py.File(hffilePCA, 'w')
hf = h5py.File(hffile, 'w')

for ss in slicename:
    slice = ss
    print('Processing '+ss +'...\n')
    flist = glob.glob(fileDir+slice+'\*.csv')
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
    transformer = StandardScaler().fit(res)

    npc = 5
    from sklearn.decomposition import PCA
    pca = PCA(n_components=npc)
    pca.fit(res.drop(columns=['PCWE']))

    print(np.cumsum(pca.explained_variance_ratio_))

    gruppo = hf2.create_group(slice)
    for aa, file in enumerate(flist):
        datum = pd.read_csv(file,names=header,skiprows=1)
        #datum['PCWE']=datum['PCWE'].astype('float')
        #datum.replace([np.inf, -np.inf], np.nan, inplace=True)
        #datum = datum.dropna()
        datum['TKE']=0.5*(datum['Ur']**2+datum['Ut']**2+datum['Ux']**2)
        datum['Theta']  = np.arctan(datum['Y']/datum['Z'])
        coords = datum[['X','Y','Z']]
        datum = datum.drop(columns=['TKE_2','PCWE_1','X','Y','Z','Theta'])
        gruppo2 = gruppo.create_group('slice_'+str(aa))
        dati = transformer.transform(datum)
        sel2 = pd.DataFrame(dati,columns=datum.columns)
        sel = sel2['PCWE']

        dati = pca.transform(sel2.drop(columns=['PCWE']))
        dati = dati[:,:npc]
        colnames = ['score_'+str(x) for x in range(npc)]
        dati = pd.concat([pd.DataFrame(dati,columns=colnames),sel,coords], axis=1)
        for jj in dati.columns:
            gruppo2[jj] = dati[jj]




    gruppo = hf.create_group(slice)
    for aa, file in enumerate(flist):
        datum = pd.read_csv(file,names=header,skiprows=1)
        #datum['PCWE']=datum['PCWE'].astype('float')
        #datum.replace([np.inf, -np.inf], np.nan, inplace=True)
        #datum = datum.dropna()
        datum['TKE']=0.5*(datum['Ur']**2+datum['Ut']**2+datum['Ux']**2)
        datum['Theta']  = np.arctan(datum['Y']/datum['Z'])
        coords = datum[['X','Y','Z']]
        datum = datum.drop(columns=['TKE_2','PCWE_1','X','Y','Z','Theta'])
        gruppo2 = gruppo.create_group('slice_'+str(aa))
        dati = transformer.transform(datum)
        dati = pd.concat([pd.DataFrame(dati,columns=datum.columns),coords], axis=1)
        for jj in dati.columns:
            gruppo2[jj] = dati[jj]

hf.close()
hf2.close()


print('end')