from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd
import numpy as np
import math

#body = ["Nos", "Nec", "Rsh", "Rel", "Rwr", "Lsh", "Lel", "Lwr", "Rey", "Ley", "Rea", "Lea"]
body = ["Nos", "Nec", "Rsh", "Lsh", "Rey", "Ley", "Rea", "Lea"]       # drop wrist, elbow

dist_dict = {}
var_dict = {}
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--rawroot', type=str, required=True, help='raw data 경로 및 이름')
args = parser.parse_args()

def cal_dis(df, nose, other):  # str input으로
    h = np.array([])
    n_x = np.array(df[nose + '_X'])
    n_y = np.array(df[nose + '_Y'])
    o_x = np.array(df[other + '_X'])
    o_y = np.array(df[other + '_Y'])
    o_s = np.array(df[other + '_Score'])

    for i in range(len(df)) :
        if o_s[i]==0:
            h = np.append(h, -1)
        else:
            c = math.sqrt((n_x[i] - o_x[i])**2 + (n_y[i] - o_y[i])**2 )
            h = np.append(h, c)
    return h

def cal_var(df, num):       # num: 몇 개로 자를지 / type(df) = Series
    i = 0
    var_lst = []
    while i <= len(df):
        if i + num <= len(df):
            temp_lst = df[i:i+num]
        else:
            temp_lst = df[i:]
        
        temp_lst = np.array(temp_lst)
        if np.isnan(temp_lst).sum() == num or np.isnan(temp_lst).sum() == (num - 1):
            var_lst.append(-1)
        else:
            var = np.nanvar(temp_lst)
            var_lst.append(var)

        i = i + num

    return var_lst
    
### -- ###
df = pd.read_pickle('data_pickle/' + args.rawroot + '.pkl')    

### 거리 구해서 dictionary로 저장
for i in body:
    dist_dict[i] = cal_dis(df, 'Nos', i)    

dist_frame = pd.DataFrame(dist_dict)
print(dist_frame)

dist_frame.drop(columns='Nos', inplace=True)

# -1을 NaN으로 변경
dist_frame = dist_frame.replace(-1, np.NaN)

### 열 별로 정규화 + variance
for i in (dist_frame.columns):
    dist_frame[i] = pd.DataFrame(StandardScaler(with_mean=True, with_std=True).fit_transform(np.array(dist_frame[i]).reshape(-1, 1)), columns=[i])
    var_dict[i] = cal_var(dist_frame[i], 50)

var_df = pd.DataFrame(var_dict)
var_df['label'] = int(args.rawroot[-1])
var_df.to_pickle('data_prepared/' + args.rawroot[:-2] + '.pkl')
