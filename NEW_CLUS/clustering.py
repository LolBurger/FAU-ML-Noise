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



fileDir = r'C:\Users\lorenzo\Desktop\\'
#hffile = fileDir + 'totalData.h5'
hffile = fileDir +'totalDataPCA.h5'


NC = [14]
listT = ['slice_'+str(x) for x in range(118)]

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
            train = df[['Score_1','Score_2','Score_3']]
            cluster.fit(train)

        # Ora faccio la predizione
        for jj in listT:
            head = [key for key in f['mid'][jj].keys()]
            df = pd.DataFrame(data=None, columns=head)
            for c in df.columns:
                df[c]=np.asarray(f['mid'][jj][c])
            train = df[['Score_1', 'Score_2', 'Score_3']]
            df['label']=cluster.predict(train)
            dictval = {col: df[col].values for col in df}
            pointsToVTK(dirs+jj, df['X'].values, df['Y'].values, df['Z'].values,data=dictval)


'''
filename = "DATA/slice_radial_complete.hdf5"
f = h5py.File(filename, "r")
slices = f.keys()
griglia = pd.read_csv('DATA/slice_grid.csv')
griglia = griglia.rename(columns={'Points:0': 'x'})
for slice in slices:
    print('Processing ' + str(slice))
    snapshots = []
    times = f[slice].keys()
    for aa, time in enumerate(times):
        datas = pd.DataFrame(data=None, columns=header)
        datokeys = f[slice][time].keys()
        for cname in datum.columns:
            if cname != 'timestep':
                datas[cname] = np.asarray(f[slice][time][cname])
        snapshots.append(np.asarray(f[slice][time]['PCWE_2']))
        datas['timestep'] = time
        datum = pd.concat([datum, datas], axis=0)
f.close()

timestep = datum['timestep'].unique()

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

batch_size = len(griglia)
subset_train = datum.drop(columns=['PCWE_2', 'timestep'])
output = datum['PCWE_2']
NC = np.arange(4, 6)

tipo = 'standard'

similarity = np.zeros((len(NC), len(timestep)))

for count, nc in enumerate(NC):
    print('Processing nc= ', nc)
    if tipo == 'standard':
        cluster = MiniBatchKMeans(n_clusters=nc, random_state=0, batch_size=batch_size)
        cluster.fit(subset_train)
        dirs = 'IMAGES/nc_' + str(nc) + '/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        else:
            shutil.rmtree(dirs)
            os.makedirs(dirs)

    elif tipo == 'diff':
        cluster = MiniBatchKMeans(n_clusters=nc, random_state=0, batch_size=batch_size, reassignment_ratio=0.1)
        cluster.fit(subset_train)
        dirs = 'IMAGES/alt_nc_' + str(nc) + '/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        else:
            shutil.rmtree(dirs)
            os.makedirs(dirs)

    # labelle = kmeans.labels_
    for countt, ts in enumerate(timestep):
        subset = datum[datum['timestep'] == ts]
        labelle = cluster.predict(subset.drop(columns=['PCWE_2', 'timestep']).values)
        sub = snapshots[countt]
        y = np.linspace(griglia['x'].min(), griglia['x'].max(), endpoint=True, num=256)
        x = np.linspace(griglia['phi'].min(), griglia['phi'].max(), endpoint=True, num=256)

        xx, yy = np.meshgrid(x, y)

        from scipy.interpolate import griddata

        points = np.zeros((len(sub), 2))
        points[:, 1], points[:, 0] = griglia['x'].values, griglia['phi'].values
        grid_z2 = griddata(points, sub, (xx, yy), method='linear')
        grid_zl2 = griddata(points, labelle, (xx, yy), method='linear')
        grid_zl2 = np.nan_to_num(grid_zl2, nan=0.0)
        grid_zl2 = np.round(grid_zl2, 0)
        dai = pd.DataFrame(data=grid_zl2, columns=[str(x) for x in range(256)])
        dai.to_csv('RESULTS/' + str(ts) + '.csv', index=False)


        # PLOTTTINGGGGGGGGGGGGGGGGGGGG

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        from matplotlib.patches import Polygon

        corners = np.zeros((4, 2))
        corners[0, :] = [1.9229, 0.0088]
        corners[1, :] = [1.9405, 0.01377]
        corners[2, :] = [2.274, -0.00918]
        corners[3, :] = [2.2537, -0.01363]

        rect = Polygon(corners, closed=True, facecolor='w', fill=True, zorder=100)

        cmap = plt.get_cmap('seismic', 11)

        ax.contour(x, y, grid_z2, colors='k', levels=np.linspace(-1., 1., 10, endpoint=True))
        ax.set_xlabel('axial_coord [m]')
        ax.set_ylabel('phi [rad]')
        cmap = plt.get_cmap('rainbow', nc)
        con2 = ax.contourf(x, y, grid_zl2, cmap=cmap, levels=np.arange(-0.5, nc + 0.5))
        ax.add_patch(rect)

        fig.colorbar(con2, ax=ax, ticks=con2.cvalues, orientation='horizontal')
        fig.savefig(dirs + str(ts) + '.png')
        plt.show(block=False)
        plt.pause(0.0001)
        plt.close()
'''