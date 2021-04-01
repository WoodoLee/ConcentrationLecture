import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# parser
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--c', type=str, help='prepared (pickle)data path and name with ".pkl"')
parser.add_argument('--nc', type=str, help='prepared (pickle)data path and name with ".pkl"')
parser.add_argument('--scaler', type=int, default=0, help='0-nonscaler 1-scaler')
parser.add_argument('--file', type=str, help='merged data')
args = parser.parse_args()

# read pickle data
if args.c and args.nc:
    df_c = pd.read_pickle('../0-data/data_prepared/'+args.c)        # ( , 5)
    df_nc = pd.read_pickle('../0-data/data_prepared/'+args.nc)      # ( , 5)
if args.file:
    df = pd.read_pickle('../0-data/data_prepared/merged/'+args.file)
    df_c = df[df['label']==1]
    df_nc = df[df['label']==0]

plt.rcParams.update({'font.size': 30})      ###
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# topX, topY, midX, midY
nbin = 50
la = ['$\sigma_{Top}^X$', '$\sigma_{Top}^Y$', '$\sigma_{Mid}^X$', '$\sigma_{Mid}^Y$']

def concatData(data):
    dataT = pd.concat([data.iloc[:, 0], data.iloc[:, 1]], ignore_index=True)
    dataM = pd.concat([data.iloc[:, 2], data.iloc[:, 3]], ignore_index=True)
    dataTotal = pd.concat([dataT, dataM])
    
    return dataT, dataM, dataTotal

def drawStd(c, nc, rangemin, rangemax, xl):
    plt.clf()
    
    plt.hist(c, range=(rangemin, rangemax), bins=nbin, color='red', alpha=0.5, label='High')
    plt.hist(nc, range=(rangemin, rangemax), bins=nbin, color='blue', alpha=0.5, label='Low')

    if xl < 2:
        ticks = np.arange(0, 501, 250)
    else:
        ticks = np.arange(0, 751, 250)
        
    plt.legend()
    plt.yticks(ticks)
    plt.xlabel(la[xl])
    plt.ylabel('Count')
    plt.grid()

    plt.show()





# drop NaN
df_c = df_c.dropna(axis=0)
df_nc = df_nc.dropna(axis=0)

#print(df_c.columns)
#print(df_nc.columns)

# draw
drawStd(df_c['Top_X'], df_nc['Top_X'], -0.001, 0.025, 0)
drawStd(df_c['Top_Y'], df_nc['Top_Y'], -0.001, 0.025, 1)
drawStd(df_c['Mid_X'], df_nc['Mid_X'], -0.001, 0.025, 2)
drawStd(df_c['Mid_Y'], df_nc['Mid_Y'], -0.001, 0.025, 3)

# print
df_cLen = []
df_ncLen = []
for i in df_c.columns[:-1]:
    df_cLen.append( len(df_c[i][df_c[i] > 0.025]) / len(df_c[i]) )
    df_ncLen.append( len(df_nc[i][df_nc[i] > 0.025]) / len(df_nc[i]) )
    
print('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t{3:0.3f}'.format(df_cLen[0], df_cLen[1], df_cLen[2], df_cLen[3]))
print('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t{3:0.3f}'.format(df_ncLen[0], df_ncLen[1], df_ncLen[2], df_ncLen[3]))

#print(df_c)
#print(df_nc)
print(df_c.describe())
print(df_nc.describe())