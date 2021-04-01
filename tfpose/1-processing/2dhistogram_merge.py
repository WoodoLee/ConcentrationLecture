import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import plotly.express as px

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--c', type=str, required=True)
parser.add_argument('--nc', type=str, required=True, help='데이터 경로')
#parser.add_argument('--label', type=int)
args = parser.parse_args()

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 90})
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40,25))
plt.subplots_adjust(wspace=0.3)
fig.suptitle('Distribution of Raw Data', fontsize=110, y=0.99)

nbin = 50

histo_c = pd.DataFrame(columns=['X', 'Y', 'Score'])
histo_nc = pd.DataFrame(columns=['X', 'Y', 'Score'])

df_c = pd.read_pickle('../0-data/data_pickle/' + args.c)      # x, y, score
df_nc = pd.read_pickle('../0-data/data_pickle/' + args.nc)

body = ["Nos", "Nec", "Rsh", "Rel", "Rwr", "Lsh", "Lel", "Lwr", "Rey", "Ley", "Rea", "Lea"]

for i in body:
    XY_c = pd.concat([df_c[i + '_X'], df_c[i + '_Y'], df_c[i + '_Score']], axis=1)
    XY_nc = pd.concat([df_nc[i + '_X'], df_nc[i + '_Y'], df_nc[i + '_Score']], axis=1)
    
    XY_c.columns = ['X', 'Y', 'Score']
    XY_nc.columns = ['X', 'Y', 'Score']
    
    histo_c = histo_c.append(XY_c)
    histo_nc = histo_nc.append(XY_nc)


histo_c.Y = - histo_c.Y
histo_nc.Y = - histo_nc.Y

histo_c = histo_c[histo_c.X != 0]
histo_c = histo_c[histo_c.Y != 0]
histo_nc = histo_nc[histo_nc.X != 0]
histo_nc = histo_nc[histo_nc.Y != 0]


# fig.show()
nbins=30

counts, xedges, yedges, im0 = axes[0].hist2d(histo_c['X'], histo_c['Y'], bins=nbins, range=[[0, 1], [-1,0]])
axes[0].set_title("Full-Concentration", fontsize=110)
axes[0].set_xlabel('X-axis')
axes[0].set_ylabel('Y-axis')

counts, xedges, yedges, im1 = axes[1].hist2d(histo_nc['X'], histo_nc['Y'], bins=nbins, range=[[0, 1], [-1,0]])
axes[1].set_title("Non-Concentration", fontsize=110)
axes[1].set_xlabel('X-axis')
axes[1].set_ylabel('Y-axis')

cb0 = fig.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.13)
cb1 = fig.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.13)

cb0.set_label('Number of entries')
cb1.set_label('Number of entries')

plt.savefig("test.png")

# python3 2dhistogram.py --file Desktop/tfpose/hci_tfpose/data_pickle/kjk_C03_1.pkl