import pandas as pd
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from reader_module import readerRes
#This code reads the results from the LES simulation in cgns format and makes
#the correlation plots for exploratory data analysis

fileDir = 'C:\\Users\Lorenzo\Desktop\DATA\\'
resDir = 'C:\\Users\Lorenzo\Desktop\IMAGES\\'
ntimes = 1
def correl_plotter(df2, fname):
    # compute correlation matrix
    corr = df2.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                           square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.text(18, 2, 'Time=' + fname + 's')
    fig = sns_plot.get_figure()
    fig.tight_layout()
    fig.savefig('IMG/CORR/' + fname + ".png", bbox_inches="tight")
    plt.close(fig)


flist = glob.glob(fileDir+'LES_1.09m3s_export_*.cgns')
for aa in range(0,ntimes):
    df, resName = readerRes(flist[aa], 'True')
    correl_plotter(df,resName)

#Computes the PCWE distribution and plots histograms for the lastest time

subdata = df[['PCWE', 'PCWE_2']]
fig,ax = plt.subplots(1,2,sharey=True)
ax[0].hist(df['PCWE'].values, density=True, bins=50, range=[-6000, 6000], log=True)
ax[1].hist(df['PCWE_2'].values,density=True, bins=50,range=[-10,10],color='r', log=True)
ax[0].set_xlabel('PCWE_Sources')
ax[1].set_xlabel('PCWE_Sources')
ax[0].set_ylabel('Probability density')
plt.savefig('IMG/CORR/PCEW_hist')
plt.close(fig)

print('End')