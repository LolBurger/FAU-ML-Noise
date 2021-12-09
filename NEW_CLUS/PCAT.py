import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import h5py

h5file = h5py.File(r'C:\Users\lorenzo\Desktop\totalData.h5','r')

slices = h5file['mid'].keys()
features = [key for key in h5file['mid']['slice_1'].keys()]
dataframe = pd.DataFrame(data=None,names=features)




h5file.close()