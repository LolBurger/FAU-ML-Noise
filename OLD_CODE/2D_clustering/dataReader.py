import pandas as pd
import numpy as np
import h5py
import glob
from sklearn.preprocessing import RobustScaler
import pickle

fileDir = 'C:\\Users\Lorenzo\Desktop\DATA\\'
resDir = 'C:\\Users\Lorenzo\Desktop\IMAGES\\radial_data\\'
sample = 1 #determines the sampling frequency

grid = pd.read_csv('C:\\Users\Lorenzo\Desktop\DATA\Mean\\gridPoints.csv')
grid['radius']=(grid['Points:2']**2+grid['Points:1']**2)**0.5
splits = [0.1875]
transformer = RobustScaler(quantile_range=[20.0, 80.0])
feat = ['PCWE_2','Velocity_0','Velocity_1', 'Velocity_2','TurbulentViscosityRatio','Pressure']
with h5py.File('DATA/slice_radial_complete.hdf5', 'w') as hf:
    for splitV in splits:
        grpG = hf.create_group(str(splitV))

    for count, file in enumerate(glob.glob(fileDir+'LES_1.09m3s_export_*.cgns')):
        if count == 120:
            break
        else:
            print('Reading:', file)
            print('Progress:', str((count + 1) / (120 + 1) * 100))
            resName = file.replace('C:\\Users\Lorenzo\Desktop\DATA\LES_1.09m3s_export_', '')
            resName = resName.replace('e+01.cgns', '')
            f = h5py.File(file, 'r')
            region = f['Base']['rotor_new']
            sol = region['Solution0000001']  # access the solution
            fields = list(sol.keys())  # saves the stored field names as a list
            fields = fields[2::]  # drops trash
            df = pd.DataFrame(data=None, columns=fields)  # create an empty df
            df.columns = df.columns.str.replace("Monitor", "")
            df = pd.concat([df, grid], axis=1)

            for splitV in splits:
                # stores the undersampled solution in the dataframe
                for cname in fields:
                    df[cname] = np.array(sol[cname][' data'][::sample])

                sub = df[df['radius'].between(splitV-0.001,splitV+0.001)]
                sub['magU'] = (sub['Velocity_0']**2+sub['Velocity_1']**2+sub['Velocity_2']**2)**0.5
                sub['phi']=np.arctan2(sub['Points:2'].values,sub['Points:1'].values)
                sub = sub[sub['WallDistance']<0.01]

                #LEVA LE COORDINATE
                if count == 0:
                    #grpp = hf[str(splitV)].create_group('Grid')
                    dfgrid = pd.concat([sub['Points:0'],sub['phi']],axis=1)
                    dfgrid.to_csv('DATA/slice_grid.csv',index=False)
                    dname = sub.columns
                    sub = transformer.fit_transform(sub)
                    pickle.dump(transformer, open('MODELS/normalizer.save', 'wb'))
                else:
                    transformer = pickle.load(open('MODELS/normalizer.save', 'rb'))
                    sub = transformer.transform(sub)

                grp = hf[str(splitV)].create_group(resName)
                #grp['PCWE']=sub['PCWE_2'].values
                sub = pd.DataFrame(data=sub,columns=dname)
                for jj in feat:
                    grp[jj]=sub[jj].values

print('end')





