import argparse
import matplotlib
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


plt.rcParams.update({'font.size': 15})      ###
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.figure(1, dpi=300)



# topX, topY, midX, midY
'''plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 70})'''
fig, axes = plt.subplots(nrows=2, ncols=2, dpi=300)          # , figsize=(30, 25)
fig.tight_layout(pad=1.4)
nbin = 50
fig.suptitle('Distribution of $\sigma_{Top}$, $\sigma_{Mid}$', y=0.99)

# label list
#la = ['Standard deviation of\ntop\'s x-coordinate', 'Standard deviation of\ntop\'s y-coordinate', 'Standard deviation of\nmid\'s x-coordinate', 'Standard deviation of\nmid\'s y-coordinate']
la = ['$\sigma_{Top}^X$', '$\sigma_{Top}^Y$', '$\sigma_{Mid}^X$', '$\sigma_{Mid}^Y$']


# define concatenation function
def concatData(data):
    dataT = pd.concat([data.iloc[:, 0], data.iloc[:, 1]], ignore_index=True)
    dataM = pd.concat([data.iloc[:, 2], data.iloc[:, 3]], ignore_index=True)
    dataTotal = pd.concat([dataT, dataM])
    
    return dataT, dataM, dataTotal

# define drawing std 1-D histogram function
def drawStdHist(data, row, rangemin,rangemax):
    if row == 1:    # concentrate
        c = 'red'      
        l = 'High'
    else:           # not concentrate
        c = 'blue'            
        l = 'Low'

    idx = 0
    for i in range(2):         
        for j in range(2):
            n, bins, patches = axes[i][j].hist(data.iloc[:, idx], range=(rangemin, rangemax), bins=nbin, alpha=0.5, color=c, label=l)
            axes[i][j].legend()
            
            if idx <= 1:        # top
                major_ticks = np.arange(0, 501, 250)
            else:               # body
                major_ticks = np.arange(0, 751, 250)
            
            axes[i][j].set_yticks(major_ticks)
            axes[i][j].set_xlabel(la[idx])
            axes[i][j].set_ylabel('Count')
            axes[i][j].grid(True, alpha=0.5)

            idx += 1
    


# drop NaN
df_c = df_c.dropna(axis=0)
df_nc = df_nc.dropna(axis=0)

# standardScaler
if args.scaler:
    standardScaler = StandardScaler()
    df = pd.concat([df_c, df_nc])
    for i in df_c.columns[:-1]:
        #df_c[i] = (df_c[i] - df_c[i].mean()) / df_c[i].std()
        #df_nc[i] = (df_nc[i] - df_nc[i].mean()) / df_nc[i].std()
        df[i] = (df[i] - df[i].mean()) / df[i].std()
    df_c = pd.DataFrame(standardScaler.fit_transform(df_c))
    df_nc = pd.DataFrame(standardScaler.fit_transform(df_nc))
    
    # print(df.describe())
    df_c = df[df['label'] == 1]
    df_nc = df[df['label'] == 0]

# draw
drawStdHist(df_c, 1, -0.001, 0.025)
drawStdHist(df_nc, 0, -0.001, 0.025)

# plt.show()
plt.savefig('kjk_stopmove_50.png')

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



# 어떤 데이터는 0.025 밖에서 많이 관측됨 (예 - mid)
# 이 그래프는 경향성만 보여준다.
# DNN의 input으로는 데이터는 다 썼다.

