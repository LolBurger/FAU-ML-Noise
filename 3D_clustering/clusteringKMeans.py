import pandas as pd
from sklearn.cluster import KMeans
import glob
import os, shutil
import matplotlib.pyplot as plt
#This code performs kMean clustering optimization based on number of clusters on normalized data
fileDir = 'C:\\Users\Lorenzo\Desktop\DATA\\'
ntimes = 1
grid = pd.read_csv('C:\\Users\Lorenzo\Desktop\DATA\Mean\gridPoints.csv')
flist = glob.glob(fileDir+'LES_1.09m3s_export_*.cgns')
resultDir = 'RESULTS/KMEANS/'

n_c = [x for x in range(2,3)]

def plotCl(data, dirIm, idx):
    snaps = []
    for cid in range(len(data.columns)-3):
        dato = data[data['cluster'] == cid]
        snaps.append(dato['PCWE_2'])

    plt.figure(figsize=(12, 9))
    plt.hist(snaps, stacked=True, bins=60, label=[str(x) for x in range(len(data.columns)-3)])
    #plt.xlim(-1300, 750)
    plt.yscale('log')
    plt.legend(loc=2, fontsize=22)
    plt.xlabel('PCWE_sources')
    plt.ylabel('log_of_occurrences')
    plt.savefig(dirIm+'/'+str(idx)+'.png')
    plt.close()

for aa in range(0,1):
    data = pd.read_csv('C:\\Users\Lorenzo\Desktop\DATA\PROCESSED/' + str(aa) + '_PCA.csv' + '.csv')
    grid = data[['Points:0', 'Points:1', 'Points:2']]
    output = data['PCWE_2']
    input = data.drop(columns=['PCWE_2','Points:0', 'Points:1', 'Points:2'])

    for nc in n_c:
        kmeans = KMeans(n_clusters=nc, random_state=52)
        kmeans.fit(input)
        xtransf=kmeans.predict(input)
        xtransf =pd.DataFrame(data=xtransf, columns=['cluster'])
        labels = kmeans.labels_
        dataset = pd.concat([xtransf,grid,output],axis=1)

        dir = 'RESULTS/KMEANS/NC='+str(nc)
        dirIm = 'IMG/KMEANS/NC='+str(nc)
        if not os.path.exists(dir):
                os.makedirs(dir)
        else:
                shutil.rmtree(dir)
                os.makedirs(dir)
        if not os.path.exists(dirIm):
                os.makedirs(dirIm)
        else:
                shutil.rmtree(dirIm)
                os.makedirs(dirIm)

        dataset.to_csv(dir+'/results'+str(aa)+'.csv',index=False)

        descript = pd.DataFrame(data=None, index=dataset.describe().index.tolist())
        for cid in range(nc):
            sotto = dataset[dataset['cluster']==cid]
            cname = 'cl'+str(cid)
            descript[cname] = sotto['PCWE_2'].describe()
        descript.to_csv(dir+'/describe'+str(aa)+'.csv')
        plotCl(dataset,dirIm,aa)


print('End')