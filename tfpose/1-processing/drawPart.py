import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# parser
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--c', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
parser.add_argument('--nc', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
parser.add_argument('--num', type=int, default=50, help='per N frames')
args = parser.parse_args()

# read pickle data
df_c = pd.read_pickle('../0-data/data_pickle/'+args.c)       
df_nc = pd.read_pickle('../0-data/data_pickle/'+args.nc)      

plt.rcParams.update({'font.size': 25})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#fig, axes = plt.subplots(nrows=1, ncols=1, dpi=300) 

# fig.tight_layout(pad=2)
# fig.suptitle('Distribution of $\sigma_{Neck}$', y=0.95)
nbin = 50

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

def drawStdHistX(datac, datanc, rangemin, rangemax):
    plt.hist(datac, range=(rangemin, rangemax), bins=nbin, color='red', alpha=0.5, label='High')
    plt.hist(datanc, range=(rangemin, rangemax), bins=nbin, color='blue', alpha=0.5, label='Low')

    plt.legend()
    plt.xlabel('$\sigma_{Neck}^X$')
    plt.ylabel('Count')
    plt.grid()

def drawStdHistY(datac, datanc, rangemin, rangemax):
    plt.hist(datac, range=(rangemin, rangemax), bins=nbin, color='red', alpha=0.5, label='High')
    plt.hist(datanc, range=(rangemin, rangemax), bins=nbin, color='blue', alpha=0.5, label='Low')

    plt.legend()
    plt.xlabel('$\sigma_{Neck}^Y$')
    plt.ylabel('Count')
    plt.grid()

c_stdX, c_stdY = getStd(df_c, args.num)
nc_stdX, nc_stdY = getStd(df_nc, args.num)

df_c = pd.concat([c_stdX, c_stdY], axis=1)
df_c['label'] = 1
df_nc = pd.concat([nc_stdX, nc_stdY], axis=1)
df_nc['label'] = 0

print(df_c)
print(df_nc)

print(df_c.describe())
print(df_nc.describe())

drawStdHistX(df_c['stdX'], df_nc['stdX'], -0.001, 0.01)
plt.show()
#plt.savefig('kjk_neck50_X.png', dpi=300)

plt.clf()
drawStdHistY(df_c['stdY'], df_nc['stdY'], -0.001, 0.01)
plt.show()
#plt.savefig('kjk_neck50_Y.png', dpi=300)
