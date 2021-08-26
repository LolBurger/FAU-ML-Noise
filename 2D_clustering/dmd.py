import numpy as np
import scipy
import scipy.integrate
from matplotlib import pyplot as plt
from pydmd import DMD, OptDMD, MrDMD
import pandas as pd
import glob
import os


resDir = 'RESULTS'
flist = sorted(os.listdir(resDir))
snapshots = []
for filen in flist:
    dato = pd.read_csv(resDir+'/'+filen)
    snapshots.append(dato.values)


valuesD = []

import shutil
import os

from matplotlib.colors import Normalize

dmd = DMD(svd_rank=0, tlsq_rank=0, exact=True, opt=True)
dmd.fit(snapshots)

dir = 'IMAGES/DMD'

dmd.plot_eigs()
plt.savefig(dir+'/eiges.png')
plt.close('all')

modes = dmd.modes
for jj in range(5):

        smode = modes[:,jj*2]
        maxa = smode.real.max()
        smode = np.reshape(smode.real,(256,256))
        fig = plt.figure(figsize=(6,6))

        from matplotlib.patches import Polygon

        corners = np.zeros((4, 2))
        corners[0, :] = [1.9229, 0.0088]
        corners[1, :] = [1.9405, 0.01377]
        corners[2, :] = [2.274, -0.00918]
        corners[3, :] = [2.2537, -0.01363]

        rect = Polygon(corners, closed=True, facecolor='w', fill=True, zorder=100)


        plt.title('Mode='+ str(jj))
        plt.set_cmap('seismic')

        plt.imshow(smode, origin='lower',vmin=-maxa, vmax=maxa,  cmap='seismic')

        #plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.tight_layout()
        plt.savefig(dir + '/mode' + str(jj) + '.png')
        plt.close(fig)


ntimes = [0,10,20,39]
for hh in ntimes:
        fs, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
        snap = dmd.snapshots
        maxsn = snap.max().real
        rec = dmd.reconstructed_data
        ax1.matshow(np.reshape(snap[:,hh].real,(256,256)), vmin=-maxsn, vmax=maxsn, origin='lower', cmap='seismic')
        ax1.set_title('original')
        dd = ax2.matshow(np.reshape(rec[:, hh].real, (256, 256)), vmin=-maxsn, vmax=maxsn, origin='lower',cmap='seismic')
        ax2.set_title('reconst')
        plt.colorbar(dd)
        plt.savefig(dir + '/data' + str(hh) + '.png')
        plt.close(fs)

for index in range(4):
        dmd.plot_modes_2D(index, figsize=(12, 5), cmap='seismic')
        plt.savefig(dir+ '/modeA' + str(index) + '.png')
        plt.close('all')


fig = plt.figure(figsize=(16, 9))
plt.plot(scipy.linalg.svdvals(np.array([snapshot.flatten() for snapshot in snapshots]).T), 'o')
plt.savefig(dir + '/svd.png')
plt.close('all')


