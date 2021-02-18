import csv
import pandas as pd
import numpy as np

files = ['kpg_ra4_1.pkl', 'kpg_ra5_0.pkl', 'kjk_stop_1.pkl', 'kjk_move_0.pkl']
csv = ['fig3-kpg_1_8000.csv', 'fig3-kpg_0_8000.csv', 'fig4-kjk_neck50-stop.csv', 'fig4-kjk_neck50-move.csv']

files2 = ['kjk_stop_50.pkl', 'kjk_move_50.pkl']
csv2 = ['fig5-kjk_stopmove_50-stop.csv', 'fig5-kjk_stopmove_50-move.csv']

# df = pd.read_pickle('../0-data/data_pickle/kpg_ra4_1.pkl')
# df = pd.read_pickle('../0-data/data_pickle/kpg_ra5_0.pkl')

for i, j in zip(files, csv):
    df = pd.read_pickle('../0-data/data_pickle/' + i)
    df.to_csv(j, mode='a')

for i, j in zip(files2, csv2):
    df = pd.read_pickle('../0-data/data_prepared/' + i)
    df.to_csv(j, mode='a')