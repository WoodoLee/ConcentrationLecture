import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy import special, optimize
from pylab import *
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline, BSpline



# plt.rcParams['font.family'] = 'Times New Roman'

plt.rcParams.update({'font.size': 11})      ###

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='x-small')
# plt.rc('ytick', labelsize='x-small')
# params = {'tex.usetex': True}
# plt.rcParams.update(params)

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"]
# })

# plt.rc(usetex = True)
dataPath0 = '~/Work/python/ConcentrationLectureData/data/test111.pkl'
dataPath1 = '~/Work/python/ConcentrationLectureData/data/test000.pkl'







def funcKFsimple(datapath):
    dfRaw = pd.read_pickle(datapath)
    # dfRaw = dfRaw[500:1700].reset_index(drop=True)

    tFrame =  50.
    tFps = 20.
    tWin = tFrame * (1/tFps)
    tMax = tWin * len(dfRaw)
    n4error = tFrame
    error = 12 / np.sqrt(n4error)

    dfMeasure = dfRaw['prediction']

    # print(dfMeasure)


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
    x_0 = 0.5  
    P_0 = 0.9

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

    return conLv_meas_save, conLv_esti_save, tNpTot


m1, e1, tNpTot1 = funcKFsimple(dataPath1)
m0, e0, tNpTot0 = funcKFsimple(dataPath0)
tNpTot1 = tNpTot1 * 2.5 / 60
tNpTot0 = tNpTot0 * 2.5 / 60

plt.figure(1, figsize=(5, 4), dpi=300)              ###

plt.plot(tNpTot1, m1, 'bo', label='Measurements 1 ', markersize=2 )
plt.plot(tNpTot1, e1, 'ro', label='Estimations 1 ' , markersize=2 )
plt.plot(tNpTot0, m0, 'ko', label='Measurements 0 ', markersize=2 )
plt.plot(tNpTot0, e0, 'go', label='Estimations 0 ' , markersize=2 )


plt.legend(framealpha=1, loc='upper right', ncol= 2)
plt.title('Results')
plt.xlabel('Time (min)')
plt.ylabel('Concentration Levels')
# plt.xlim(0,50)
# plt.ylim(0,1.0)
plt.grid()
plt.savefig('./png/result.png')


def _2gaussian(x, amp1,cen1,sigma1, amp2,cen2,sigma2):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen2)/sigma2)**2)))

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def min_max_normalize(lst):
    normalized = []
    
    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst) - min(lst))
        normalized.append(normalized_num)
    
    return normalized

def z_score_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)
    return normalized



plt.figure(2,figsize=(5, 4), dpi=300)
# plt.hist(conLv_esti_save, bins =nBins , label='Estimation')
# plt.legend(loc='lower right')


# e1 = z_score_normalize(e1)
# e0 = z_score_normalize(e0)

nBins0 = 30

stdIni0 = np.std(e0)
meanIni0 = np.mean(e0)

y0,x0,_ = hist(e0, bins =nBins0 , label='Estimation', color='green', alpha=0.5, range=(0.0,0.25))

x0=(x0[1:]+x0[:-1])/2 # for len(x)==len(y)

expected0=(meanIni0 - stdIni0, stdIni0/2 , 50, meanIni0 + stdIni0  , stdIni0/2, 100)

params0,cov0 = curve_fit(bimodal,x0,y0,expected0)

sigma0=sqrt(diag(cov0))

#define x as 200 equally spaced values between the min and max of original x 
xnew0 = np.linspace(x0.min(), x0.max(), 200) 

#define spline
spl0 = make_interp_spline(x0, bimodal(x0,*params0), k=3)
y_smooth0 = spl0(xnew0)

plot(xnew0,y_smooth0,color='green',lw=3,label='model')

print(pd.DataFrame(data={'params0':params0,'sigma0':sigma0},index=bimodal.__code__.co_varnames[1:]))

plt.title('Low Concentration')
plt.xlabel('$\Psi_{low}$')
plt.ylabel('Number of Values') 
plt.grid()
plt.savefig('./png/resultLow.png')

#############


plt.figure(3,figsize=(5, 4), dpi=300)
# plt.hist(conLv_esti_save, bins =nBins , label='Estimation')
# plt.legend(loc='lower right')


nBins1 = 30

stdIni1 = np.std(e1)
meanIni1 = np.mean(e1)

y1,x1,_ = hist(e1, bins =nBins1 , label='Estimation', color='red', alpha=0.5, range=(0.7,0.85))

x1=(x1[1:]+x1[:-1])/2 # for len(x)==len(y)

expected1=(0.75, stdIni1/2 , 120, meanIni1 + stdIni1  , stdIni1/2, 100)

params1,cov1 = curve_fit(bimodal,x1,y1,expected1)

sigma1=sqrt(diag(cov1))

#define x as 200 equally spaced values between the min and max of original x 
xnew1 = np.linspace(x1.min(), x1.max(), 200) 

#define spline
spl1 = make_interp_spline(x1, bimodal(x1,*params1), k=3)
y_smooth1 = spl1(xnew1)

plot(xnew1,y_smooth1,color='red',lw=3,label='model')

print(pd.DataFrame(data={'params1':params1,'sigma1':sigma1},index=bimodal.__code__.co_varnames[1:]))

plt.title('High Concentration')
plt.xlabel(r'$\Psi_{high}$')
plt.ylabel('Number of Values')
plt.grid()
plt.savefig('./png/resultHigh.png')




plt.show()
