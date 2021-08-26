import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
#This code reads experimental measurements from FAU (Felix) and computes
# the Reynolds stress tensor and length scale for Synthetic Eddy Method
# (inflow for LES)

wd = 'C:/Users/Lorenzo/Desktop/FAU/'
filename = wd + 'HDA_3D_TG1_14m3s_Lorenzo_AlteMessung_angepasst.mat'
arrays = []
#read the file
f = h5py.File(filename,'r')
data = f.get('data/variable1')

def plotter(cnamess, data):
    for con, jj in enumerate(cnamess):
        if con > 1:
            plt.figure()
            ax = plt.subplot(111)
            ax.set_aspect(1)
            cb = ax.scatter(data['Y'],data['Z'],c=data[jj])
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax.set_title(jj)
            plt.colorbar(cb)
            plt.tight_layout()
            plt.savefig('IMG_inflow/'+jj+'.png',bbox_inches='tight')
            plt.close()


#find the keys
chiavi = f.keys()
Ut = np.zeros((80))

#cnames = ['II', 'III', 'TuGlo', 'TuLok', 'UV', 'UW', 'U_t', 'Umean', 'Uquad', 'Urms', 'Uturb', 'VW', 'V_t', 'Vmean', 'Vquad', 'Vrms', 'Vturb', 'W_t', 'Wmean', 'Wquad', 'Wrms', 'Wturb', 'X', 'Y', 'Z', 'aniT', 'k', 'qquad', 'reyT']

#I read the nested structure
for k, v in f.items():
    for s, t in v.items():
        arrays.append(t)

arrays = arrays[:-2]
coords = np.zeros((80,2))
npf = np.zeros((80,12))

#I read the numerical values in each of the items I found and store them in another dataframe
for count, sam in enumerate(arrays):
    sample = pd.DataFrame(sam.items())
    npf[count,0]=sample[1].iloc[23][0,0]
    npf[count,1] = sample[1].iloc[24][0,0]
    npf[count, 2] = sample[1].iloc[28][0, 0]
    npf[count, 3] = sample[1].iloc[28][0, 1]
    npf[count, 4] = sample[1].iloc[28][0, 2]
    npf[count, 5] = sample[1].iloc[28][1, 0]
    npf[count, 6] = sample[1].iloc[28][1, 1]
    npf[count, 7] = sample[1].iloc[28][1, 2]
    npf[count, 8] = sample[1].iloc[28][2, 0]
    npf[count, 9] = sample[1].iloc[28][2, 1]
    npf[count, 10] = sample[1].iloc[28][2, 2]
    npf[count, 11] = sample[1].iloc[2][0, 0]

cnames=['Y','Z','R_00','R_01','R_02','R_10','R_11','R_12','R_20','R_21','R_22','LS']
npf = pd.DataFrame(data=npf,columns=cnames)
npf['radius']=(npf['Y']**2+npf['Z']**2)**0.5
npfr = npf.groupby(by='radius',as_index=False).mean()

plotter(cnames,npfr)

npfr.to_csv(wd+'distributed_data.csv',index=False)

npfr2 = npfr.mean()
npfr2.to_csv('Temporary_files/averaged_data.csv')