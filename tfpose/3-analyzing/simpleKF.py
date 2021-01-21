import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy import special, optimize
from pylab import *
from scipy.optimize import curve_fit

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 40})

dataPath = '../0-data/data_prediction/4R16R1S_es_100/90_for_kf.pkl'

dfRaw = pd.read_pickle(dataPath)

print(dfRaw)


tFrame =  50.
tFps = 20.
tWin = tFrame * (1/tFps)
tMax = tWin * len(dfRaw)


n4error = tFrame

error = 12 / np.sqrt(n4error)



print(error)

dfMeasure = dfRaw['prediction']

print(dfMeasure)


dfMeasureCut = pd.DataFrame() 

nCut = 1

for i in range(0, len(dfRaw), nCut) :
    dfMeasureOne =  pd.DataFrame([dfMeasure.loc[i]])
    dfMeasureCut = pd.concat([dfMeasureCut, dfMeasureOne])

# print(dfMeasureCut)



tWinCut = nCut * tWin

tNp = np.arange(0, tMax, step=tWinCut)



def kalman_filter(z_meas, x_esti, P):
    """Kalman Filter Algorithm for One Variable."""
    # (1) Prediction.
    x_pred = A * x_esti
    P_pred = A * P * A + Q

    # (2) Kalman Gain.
    K = P_pred * H / (H * P_pred * H + R)

    # (3) Estimation.
    x_esti = x_pred + K * (z_meas - H * x_pred)

    # (4) Error Covariance.
    P = P_pred - K * H * P_pred

    return x_esti, P


# Initialization for system model.
A = 1
H = 1
Q = 0
R = 4
# Initialization for estimation.
x_0 = 0.5  # 14 for book.
P_0 = 6

conLv_meas_save = np.zeros(len(dfRaw))
conLv_esti_save = np.zeros(len(dfRaw))

print(len(dfRaw))
tNpTot = np.arange(0, len(dfRaw))
x_esti, P = None, None
for i in range(0, len(dfRaw), nCut):
    z_meas = dfRaw.iloc[i]
    z_meas = z_meas['prediction']

    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = kalman_filter(z_meas, x_esti, P)

    conLv_meas_save[i] = z_meas
    conLv_esti_save[i] = x_esti

plt.figure(1)
plt.plot(tNpTot, conLv_meas_save, 'ko--', label='Measurements')
plt.plot(tNpTot, conLv_esti_save, 'ro-', label='Kalman Filter')
plt.legend(loc='upper right')
plt.title('Simple Kalman Filter Result')
plt.xlabel('Time [per 2.5 sec]')
plt.ylabel('Concentration Level')
plt.savefig('./png/simple_kalman_filter.png')
plt.grid()

nBins = 150
def _2gaussian(x, amp1,cen1,sigma1, amp2,cen2,sigma2):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen2)/sigma2)**2)))



def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)





# popt_2gauss, pcov_2gauss = optimize.curve_fit(_2gaussian, x_array, y_array_2gauss, p0=[amp1, cen1, sigma1, amp2, cen2, sigma2])
# perr_2gauss = np.sqrt(np.diag(pcov_2gauss))
# pars_1 = popt_2gauss[0:3]
# pars_2 = popt_2gauss[3:6]
# gauss_peak_1 = _1gaussian(x_array, *pars_1)
# gauss_peak_2 = _1gaussian(x_array, *pars_2)



plt.figure(2)
# plt.hist(conLv_esti_save, bins =nBins , label='Estimation')
plt.legend(loc='upper right')
plt.title('Estimation Histogram')
plt.xlabel('Estimation Concentration Level')
plt.ylabel('Number of Values')
# plt.savefig('./png/histogram.png')
plt.grid()


stdIni = np.std(conLv_esti_save)
meanIni = np.mean(conLv_esti_save)
y,x,_ = hist(conLv_esti_save, bins =nBins , label='Estimation', color='blue', alpha=0.5)

x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

expected=(0.48, stdIni , 50, 0.52 , stdIni, 20)
params,cov=curve_fit(bimodal,x,y,expected)
sigma=sqrt(diag(cov))
plot(x,bimodal(x,*params),color='red',lw=3,label='model')
legend()
# print(params,'\n',sigma)   
# print(pd.DataFrame(data={'params':params,'sigma':sigma},index=bimodal.__code__.co_varnames[1:]))

plt.show()
