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
from sklearn.mixture import GaussianMixture
from pyevtk.hl import pointsToVTK
from scipy.spatial.distance import cdist


trainS = 0   #1 se PCA 1 / 0 variabili tal quali


fileDir = r'C:\Users\Lorenzo\Desktop\dataexp\\'
#hffile = fileDir + 'totalData.h5'
if trainS == 1:
    hffile = fileDir +'totalDataPCA.h5'
else:
    hffile = fileDir +'totalData.h5'



eps = [1,2,3,4,6,8,10,14,18,20]
cov = ['full','tied']
#listT = ['slice_'+str(x) for x in range(118)]
listT = ['slice_10']
slices = ['hub', 'mid', 'tip']



with h5py.File(hffile, 'r') as f:
    for ss in slices:
        emptyData = pd.DataFrame(data=None, columns=None)
        for nc in eps:
            for ns in cov:

                cluster = GaussianMixture(n_components=nc, covariance_type=ns)
                for jj in listT:
                    head = [key for key in f[ss][jj].keys()]
                    df = pd.DataFrame(data=None, columns=head)
                    for c in df.columns:
                        df[c]=np.asarray(f[ss][jj][c])
                    #if ss == 'tip':
                    #    df = df.sample(frac=0.1)
                    train = df.drop(columns=['PCWE','X','Y','Z'])

                    dname = 'IDS_nmixture_'+str(nc)+'_cov_'+str(ns)
                    print('Processing '+ss+' '+dname)
                    emptyData[dname]=cluster.fit_predict(train)

        df = pd.concat([df[['PCWE','X','Y','Z']],emptyData],axis=1)
        dictval = {col: df[col].values for col in df}
        pointsToVTK('RESULTS/GM_PCA_'+str(trainS)+'_'+ss, df['X'].values, df['Y'].values, df['Z'].values,data=dictval)


print('END')