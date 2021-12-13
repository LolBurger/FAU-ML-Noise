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
from sklearn.cluster import MiniBatchKMeans
from pyevtk.hl import pointsToVTK
from scipy.spatial.distance import cdist



#fileDir = r'C:\Users\lorenzo\Desktop\\'
fileDir = r'C:\Users\Utente\Desktop\dataexp\\'
#hffile = fileDir + 'totalData.h5'
hffile = fileDir +'totalDataPCA.h5'
hffile = fileDir +'totalData.h5'



distortions = []
inertias = []
mapping1 = {}
mapping2 = {}


NC = np.arange(2,61,1,dtype=int)
NC = [18]
#listT = ['slice_'+str(x) for x in range(118)]
listT = ['slice_10']

with h5py.File(hffile, 'r') as f:
    for nc in  NC:
        cluster = MiniBatchKMeans(n_clusters=nc, random_state=10, batch_size=8940)
        dirs = 'RESULTS/nc_' + str(nc) + '/'
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
            #train = df[['Score_1','Score_2','Score_3']]
            train = df.drop(columns=['PCWE','X','Y','Z'])
            cluster.fit(train)
            distortions.append(sum(np.min(cdist(train, cluster.cluster_centers_,
                                                'euclidean'), axis=1)) / train.shape[0])
            inertias.append(cluster.inertia_)

            mapping1[nc] = sum(np.min(cdist(train, cluster.cluster_centers_,
                                           'euclidean'), axis=1)) / train.shape[0]
            mapping2[nc] = cluster.inertia_

        # Ora faccio la predizione
        for jj in listT:
            head = [key for key in f['mid'][jj].keys()]
            df = pd.DataFrame(data=None, columns=head)
            for c in df.columns:
                df[c]=np.asarray(f['mid'][jj][c])
            #train = df[['Score_1', 'Score_2', 'Score_3']]
            train = df.drop(columns=['PCWE', 'X', 'Y', 'Z'])
            df['label']=cluster.predict(train)
            dictval = {col: df[col].values for col in df}
            pointsToVTK(dirs+jj, df['X'].values, df['Y'].values, df['Z'].values,data=dictval)

'''
plt.figure()
plt.plot(NC, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.savefig('IMAGES/distortion_elbow_chart.png')
plt.show()

plt.figure()
plt.plot(NC, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.savefig('IMAGES/inertia_elbow_chart.png')
plt.show()
'''

print('END')