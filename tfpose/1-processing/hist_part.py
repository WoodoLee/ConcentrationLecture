import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# parser
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--c', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
parser.add_argument('--nc', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
parser.add_argument('--scaler', type=int, default=0, help='0-nonscaler 1-scaler')
parser.add_argument('--num', type=int, default=100, help='per N frames')
args = parser.parse_args()

# read pickle data
df_c = pd.read_pickle('../0-data/data_pickle/'+args.c)       
df_nc = pd.read_pickle('../0-data/data_pickle/'+args.nc)      

# neckx, necky
fig, axes = plt.subplots(nrows=1, ncols=2)      
nbin = 50
font = {'family': 'Times New Roman',
        'size': 40,
        }
fig.suptitle('Distribution of Neck', fontproperties=font)

part = 'Nec'

df_c = df_c[[part + '_X', part + '_Y']]
df_nc = df_nc[[part + '_X', part + '_Y']]

def getStd(data, num):
    stdX = np.array([])
    stdY = np.array([])
    lend = len(data)
    idx = 0
    while (idx < lend):
        tmp = data[idx:idx+num]
        lent = len(tmp)
        if (lent > 1):
            stdX = np.append(stdX, (np.std(tmp[part + '_X'])))
            stdY = np.append(stdY, (np.std(tmp[part + '_Y'])))
            idx += num
        else:
            break

    return pd.DataFrame(stdX, columns=['stdX']), pd.DataFrame(stdY, columns=['stdY'])
    
# define drawing std 1-D histogram function
def drawStdHist(dataX, dataY, row, rangemin, rangemax):
    for i in range(2):
        if i == 0:
            data = dataX
            label = part + '_X'
            
        else:
            data = dataY
            label = part + '_Y'
        
        if row==1: c='red'      # concentration
        else: c='blue'          # not

        axes[i].hist(data, range=(rangemin, rangemax), bins=nbin, color=c, alpha=0.5, label='label: ' + str(row))
        axes[i].legend(prop=font)
        axes[i].set_xlabel(label, fontdict=font)
        axes[i].set_ylabel('Num', fontdict=font)

c_stdX, c_stdY = getStd(df_c, args.num)
nc_stdX, nc_stdY = getStd(df_nc, args.num)

df_c = pd.concat([c_stdX, c_stdY], axis=1)
df_c['label'] = 1
df_nc = pd.concat([nc_stdX, nc_stdY], axis=1)
df_nc['label'] = 0

if args.scaler:
    df = pd.concat([df_c, df_nc])
    
    for i in df.columns[:-1]:
        df[i] = (df[i] - df[i].mean()) / df[i].std()    # normalize
        # df[i] = df[i] / (df[i].max() - df[i].mean())     # minmax

    df_c = df[df['label'] == 1]
    df_nc = df[df['label'] == 0]

print(df_c)
print(df_nc)

print(df_c.describe())
print(df_nc.describe())

drawStdHist(df_c['stdX'], df_c['stdY'], 1, 0, 0.02)        # concentrate
drawStdHist(df_nc['stdX'], df_nc['stdY'], 0, 0, 0.02)      # not concentrate

plt.show()

# print
df_cLen = []
df_ncLen = []
for i in df_c.columns[:-1]:
    df_cLen.append( len(df_c[i][df_c[i] > 0.02]) / len(df_c[i]))
    df_ncLen.append( len(df_nc[i][df_nc[i] > 0.02]) / len(df_nc[i]))
    
print('{0:0.3f}\t{1:0.3f}'.format(df_cLen[0], df_cLen[1]))
print('{0:0.3f}\t{1:0.3f}'.format(df_ncLen[0], df_ncLen[1]))

'''0.000   0.000        # percent
0.005   0.006'''
'''0.000   1.000        # the number of
10.000  14.000'''
