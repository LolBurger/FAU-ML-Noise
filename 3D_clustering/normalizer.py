import pandas as pd
import glob
from reader_module import readerRes
from sklearn.preprocessing import RobustScaler
import pickle
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np

#This code reads and normalize data and perform PCA decomposition on request
fileDir = 'C:\\Users\Lorenzo\Desktop\DATA\\'
ntimes = 1
grid = pd.read_csv('C:\\Users\Lorenzo\Desktop\DATA\Mean\gridPoints.csv')
flist = glob.glob(fileDir+'LES_1.09m3s_export_*.cgns')
PCA = 'True'

def PCAPlotter(comps, idx, n_c):
    plt.figure(figsize=(10,10))
    im = plt.imshow(comps,vmin=-1,vmax=1)
    plt.title('Time='+str(idx))
    plt.xlabel('Feature')
    plt.ylabel('Component')
    plt.yticks(np.arange(0,n_c))
    plt.xticks(np.arange(0, len(df_norm.columns)))
    plt.colorbar(orientation='horizontal')
    plt.savefig('IMG/PCA/' + str(idx) + "_pcam.png", bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(12,9))
    plt.plot(np.arange(0, n_c), np.cumsum(pca.explained_variance_ratio_), c='k')
    plt.scatter(np.arange(0,n_c),np.cumsum(pca.explained_variance_ratio_),c='k', edgecolors='k',marker='o', ls='-')
    plt.hlines(0.95,0,n_c,ls='--',colors='k')
    plt.title('Time='+str(idx))
    plt.xlabel('N of components')
    plt.ylabel('Cumulative explained variance')
    plt.ylim(0.5,1.01)
    plt.xlim(1,n_c)
    plt.grid()
    plt.xticks(np.arange(0,n_c))
    plt.savefig('IMG/PCA/' + str(idx)  + "expv.png", bbox_inches="tight")
    plt.close(fig)


for aa in range(0,ntimes):
    data, resName = readerRes(flist[aa], 'False')


    data = data.drop(columns=data.columns[len(data.columns)-3:])
    data['Points:0'] =grid['Points:0'].values
    data['Points:1'] = grid['Points:1'].values
    data['Points:2']=grid['Points:2'].values
    #finalDB = data.drop(columns=['WallDistance'])
    df = data.sample(frac=0.05)
    griglia = df[['Points:0','Points:1','Points:2']]
    griglia = griglia.reset_index(drop=True)
    df = df.drop(columns=['Points:0','Points:1','Points:2'])
    #finalDB.to_csv('C:\\Users\Lorenzo\Desktop\DATA\PROCESSED/'+str(aa)+'_trainData.csv',index=False)
    transformer = RobustScaler(unit_variance=True, quantile_range=[20.0, 80.0])

    #START NORMALIZATION
    normname = 'MODEL/normalizer_model.save'

    if aa == 0:
        transformer.fit(df)
        pickle.dump(transformer, open(normname, 'wb'))
        print('Normalizer saved to:', normname)
    else:
        print('Normalizer loaded from:', normname)
        transformer = pickle.load(open(normname, 'rb'))

    df_norm2 = transformer.transform(df)
    df_norm2 = pd.DataFrame(data=df_norm2,columns=df.columns)
    description = df_norm2.describe()
    df_norm = df_norm2.drop(columns=['PCWE', 'PCWE_2'])
    if PCA == 'True':
        n_c=10
        pca = decomposition.PCA(n_components=n_c)
        pcaname = 'MODEL/pca_model.save'
        if aa == 0:
            pca.fit(df_norm)
            pickle.dump(pca, open(pcaname, 'wb'))
            print('PCA saved to:', pcaname)
        else:
            print('PCA loaded from:', pcaname)
            pca = pickle.load(open(pcaname, 'rb'))
        transformed_df = pca.transform(df_norm)
        exp_var_s = pca.explained_variance_ratio_
        comps = pca.components_
        tdfnames = ['score'+str(x) for x in range(n_c)]
        transformed_df = pd.DataFrame(data=transformed_df, columns=tdfnames)
        transformed_df = pd.concat([transformed_df,df_norm2['PCWE_2'],griglia],axis=1)
        transformed_df.to_csv('C:\\Users\Lorenzo\Desktop\DATA\PROCESSED/'+str(aa)+'_PCA.csv'+ '.csv',index=False)
        PCAPlotter(comps,aa,n_c)

print('\nEnd')