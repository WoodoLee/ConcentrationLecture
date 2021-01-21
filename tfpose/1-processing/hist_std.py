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
args = parser.parse_args()

# read pickle data
df_c = pd.read_pickle('../0-data/data_prepared/'+args.c)        # ( , 5)
df_nc = pd.read_pickle('../0-data/data_prepared/'+args.nc)      # ( , 5)

# topX, topY, midX, midY
fig, axes = plt.subplots(nrows=2, ncols=4)      
nbin = 15

# define concatenation function
def concatData(data):
    dataT = pd.concat([data.iloc[:, 0], data.iloc[:, 1]], ignore_index=True)
    dataM = pd.concat([data.iloc[:, 2], data.iloc[:, 3]], ignore_index=True)
    dataTotal = pd.concat([dataT, dataM])
    
    return dataT, dataM, dataTotal

#fitting Gauss
def funcFitGaus(dfInput):       # dfInput: dataframe 
    mu, std = norm.fit(dfInput)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    return x, p, mu, std

# define drawing std 1-D histogram function
def drawStdHist(data, row, rangemin,rangemax):
    # dataT, dataM, dataTotal = concatData(data)    
    
    for i in range(4):
        axes[row, i].hist(data.iloc[:, i], range=(rangemin, rangemax), bins=nbin)
        #x2, p2, mu2, std1 = funcFitGaus(data.iloc[:, i])
        #axes[row, i].plot(x2, p2, 'r', linewidth=2)

        axes[row, i].set_xlabel(data.columns[i] + '_' + str(row), fontsize=10)
        axes[row, i].set_ylabel('Num', fontsize=10)
    
    '''axes[row, 4].hist(dataT, range=(0, 1), bins=nbin)
    axes[row, 4].set_xlabel('Top_' + str(row), fontsize=10)
    axes[row, 4].set_ylabel('Num', fontsize=10)

    axes[row, 5].hist(dataM, range=(0, 1), bins=nbin)
    axes[row, 5].set_xlabel('Mid_' + str(row), fontsize=10)
    axes[row, 5].set_ylabel('Num', fontsize=10)
    
    axes[row, 6].hist(dataTotal, range=(0, 1), bins=nbin)
    axes[row, 6].set_xlabel('Total_' + str(row), fontsize=10)
    axes[row, 6].set_ylabel('Num', fontsize=10)'''



# drop NaN
df_c = df_c.dropna(axis=0)
df_nc = df_nc.dropna(axis=0)

# print before scaler
# print(df_c.describe())
# print(df_nc.describe())

# 기존에 했던 문제가 전체를 그냥 정규화해버림
# 그래서 각 colum마다 정규화하니까, describe로는 잘 보이는데 눈에는 잘 안 보임ㅠㅠ
# standardScaler
if args.scaler:
    standardScaler = StandardScaler()
    df = pd.concat([df_c, df_nc])
    for i in df.columns[:-1]:
        df[i] = (df[i] - df[i].mean()) / df[i].std()
        # df_c[i] = (df_c[i] - df_c[i].mean()) / df_c[i].std()
        # df_nc[i] = (df_nc[i] - df_nc[i].mean()) / df_nc[i].std()
    #df_c = pd.DataFrame(standardScaler.fit_transform(df_c))
    #df_nc = pd.DataFrame(standardScaler.fit_transform(df_nc))
    
    # print(df.describe())
    df_c = df[df['label'] == 1]
    df_nc = df[df['label'] == 0]

print(df_c)
print(df_nc)

print(df_c.describe())
print(df_nc.describe())

# draw
drawStdHist(df_c, 1, -5, 5)
drawStdHist(df_nc, 0, -5, 5)

plt.show()
