import pandas as pd
import numpy as np
import hdbscan
import pickle
from pyevtk.hl import pointsToVTK
import glob

def pyPoints2VTK(gridC,fields,fieldsName, fname):
        x, y, z = gridC[:,0], gridC[:,1], gridC[:,2]
        dataD = {}
        for cname in fieldsName:
                dataD[cname] = fields[cname].values
        pointsToVTK(fname, x, y, z, data = dataD)

#This code performs hdbscan clustering optimization based on cluster density on normalized data
fileDir = 'C:\\Users\Lorenzo\Desktop\DATA\\'
ntimes = 1
grid = pd.read_csv('C:\\Users\Lorenzo\Desktop\DATA\Mean\gridPoints.csv')
flist = glob.glob(fileDir+'LES_1.09m3s_export_*.cgns')
resultDir = 'RESULTS/KMEANS/'
cl_size = [1000,10000,30000]
metrica = 'euclidean'

for aa in range(0,1):
    data = pd.read_csv('C:\\Users\Lorenzo\Desktop\DATA\PROCESSED/' + str(aa) + '_PCA.csv' + '.csv')
    grid = data[['Points:0', 'Points:1', 'Points:2']]
    output = data['PCWE_2']
    input = data.drop(columns=['PCWE_2', 'Points:0', 'Points:1', 'Points:2'])
    for size in cl_size:
        clusterer = hdbscan.HDBSCAN(metric=metrica,min_cluster_size=size, prediction_data=True)
        clusterer.fit(input)
        params = clusterer.get_params()
        fname = 'alg='+str(params['algorithm'])+',metr='+str(params['metric'])+',min_cl_='\
        +str(params['min_cluster_size'])+',cl_sel_meth='+str(params['cluster_selection_method'])
        pickle.dump(clusterer, open('MODEL/HDBSCAN/'+fname+'.sav', 'wb'))
        labels = clusterer.labels_
        labelle = pd.DataFrame(data=labels, columns=['cluster'])
        dataset = pd.concat([input,labelle,grid,output],axis=1)
        dataset.to_csv('RESULTS/HDBSCAN/'+fname+'new_.csv')

        print(labels.max())
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        xtransf=clusterer.predict(input)
        xtransf =pd.DataFrame(data=xtransf, columns=['cluster'])
        resultDir = 'RESULT/HDBSCAN/'
        dataset = pd.concat([xtransf, grid, output], axis=1)




        dataset.to_csv(resultDir+'/full'+str(nc)+'.csv')
        descript = pd.DataFrame(data=None, index=dataset.describe().index.tolist())
        nc = clusterer.unique()
        for cid in range(nc):
            sotto = dataset[dataset['cluster']==cid]
            cname = 'cl'+str(cid)
            descript[cname] = sotto['PCWE'].describe()
        dataset.to_csv(resultDir+'/describe'+str(nc)+'.csv')

