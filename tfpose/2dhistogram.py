import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import plotly.express as px

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, required=True, help='데이터 경로')
args = parser.parse_args()

histo = pd.DataFrame(columns=['X', 'Y', 'Score'])

df = pd.read_pickle(args.file)      # x, y, score

body = ["Nos", "Nec", "Rsh", "Rel", "Rwr", "Lsh", "Lel", "Lwr", "Rey", "Ley", "Rea", "Lea"]

for i in body:
    
    XY = pd.concat([df[i + '_X'], df[i + '_Y'], df[i + '_Score']], axis=1)
    XY.columns = ['X', 'Y', 'Score']
    histo = histo.append(XY)

histo.Y = - histo.Y

histo = histo[histo.X != 0]
histo = histo[histo.Y != 0]

print(histo)
fig = px.density_heatmap(histo, x="X", y="Y", marginal_x="histogram", marginal_y="histogram",nbinsx=50, nbinsy=50, font='Times', fontsize=15)
fig.show()

# python3 2dhistogram.py --file Desktop/tfpose/hci_tfpose/data_pickle/kjk_C03_1.pkl


