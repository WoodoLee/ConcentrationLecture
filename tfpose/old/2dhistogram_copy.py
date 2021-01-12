import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import plotly.express as px
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, required=True, help='데이터 경로')
args = parser.parse_args()

histo = pd.DataFrame(columns=['X', 'Y', 'Score'])

df = pd.read_pickle(args.file)      # x, y, score
#df = df.iloc[:300]

npx = np.array([])
npy = np.array([])

print(np)
#body = ["Nos", "Nec", "Rsh", "Rel", "Rwr", "Lsh", "Lel", "Lwr", "Rey", "Ley", "Rea", "Lea"]
body = ["Nos", "Nec", "Rsh", "Lsh", "Rey", "Ley", "Rea", "Lea"]
#body = ['Rey', 'Ley', 'Rea', 'Lea']
for i in body:
    npx = np.append(npx, (pd.DataFrame(StandardScaler().fit_transform(np.array(df[i+'_X']).reshape(-1, 1)), columns=[i])).to_numpy())
    npy = np.append(npy, (pd.DataFrame(StandardScaler().fit_transform(np.array(df[i+'_Y']).reshape(-1, 1)), columns=[i])).to_numpy())
    # 정규화 했고

'''for i in body:    
    XY = pd.concat([df[i + '_X'], df[i + '_Y'], df[i + '_Score']], axis=1)
    XY.columns = ['X', 'Y', 'Score']
    histo = histo.append(XY)

histo.Y = - histo.Y'''
npy = -npy

#histo = histo[histo.X != 0]
#histo = histo[histo.Y != 0]

histo = pd.DataFrame({'X': npx, 'Y': npy})
print(histo)

fig = px.density_heatmap(histo, x="X", y="Y", marginal_x="histogram", marginal_y="histogram",nbinsx=100, nbinsy=100, range_x=(-1, 1), range_y=(-1, 1))
fig.update_layout(
    font=dict(
        family="Times",
        size=25)
)
fig.show()



"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(histo['X'], histo['Y'], bins=50)

plt.title('3D histogram of 2D normally distributed data points')
plt.xlabel('x axis')
plt.ylabel('y axis')

xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

plt.show()
"""

# python3 2dhistogram.py --file Desktop/tfpose/hci_tfpose/data_pickle/kpg_pnp_0.pkl
# python3 2dhistogram.py --file Desktop/tfpose/hci_tfpose/data_pickle/kpg_ml_1.pkl
