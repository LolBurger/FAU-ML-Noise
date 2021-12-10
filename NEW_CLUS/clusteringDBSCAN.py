import numpy as np
import scipy
import scipy.integrate
import pandas as pd
import glob
import os
import h5py
import shutil
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.cluster import DBSCAN
from pyevtk.hl import pointsToVTK
from scipy.spatial.distance import cdist


trainS = 0   #1 se PCA 1 / 0 variabili tal quali

#fileDir = r'C:\Users\lorenzo\Desktop\\'
fileDir = r'C:\Users\Utente\Desktop\dataexp\\'
#hffile = fileDir + 'totalData.h5'
if trainS == 1:
    hffile = fileDir +'totalDataPCA.h5'
else:
    hffile = fileDir +'totalData.h5'



eps = [10, 5, 1, 0.5, 0.1]
#listT = ['slice_'+str(x) for x in range(118)]
listT = ['slice_10']

with h5py.File(hffile, 'r') as f:
    for nc in eps:
        cluster = DBSCAN(eps=nc, min_samples=6)
        dirs = 'RESULTS_DBSCAN/eps_' + str(nc) + '/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        else:
            shutil.rmtree(dirs)
            os.makedirs(dirs)
        #Here I train incrementally on the dataset
        for jj in listT:
            head = [key for key in f['mid'][jj].keys()]
            df = pd.DataFrame(data=None, columns=head)
            for c in df.columns:
                df[c]=np.asarray(f['mid'][jj][c])
            if trainS == 1:
                train = df[['Score_1','Score_2','Score_3']]
            else:
                train = df.drop(columns=['PCWE','X','Y','Z'])
            cluster.fit(train)


        # Ora faccio la predizione
        for jj in listT:
            head = [key for key in f['mid'][jj].keys()]
            df = pd.DataFrame(data=None, columns=head)
            for c in df.columns:
                df[c]=np.asarray(f['mid'][jj][c])
            if trainS == 1:
                train = df[['Score_1','Score_2','Score_3']]
            else:
                train = df.drop(columns=['PCWE','X','Y','Z'])
            df['label']=cluster.fit_predict(train)
            dictval = {col: df[col].values for col in df}
            pointsToVTK(dirs+jj, df['X'].values, df['Y'].values, df['Z'].values,data=dictval)


print('END')