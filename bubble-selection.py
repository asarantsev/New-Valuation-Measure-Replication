import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data, label):
    print(label + ' analysis of residuals normality')
    print('Skewness:', stats.skew(data))
    print('Kurtosis:', stats.kurtosis(data))
    print('Shapiro-Wilk p = ', stats.shapiro(data)[1])
    print('Jarque-Bera p = ', stats.jarque_bera(data)[1])
    print('Autocorrelation function analysis for ' + label)
    L1orig = sum(abs(acf(data, nlags = 5)[1:]))
    print('\nL1 norm original residuals ', round(L1orig, 3), label, '\n')
    L1abs = sum(abs(acf(abs(data), nlags = 5)[1:]))
    print('L1 norm absolute residuals ', round(L1abs, 3), label, '\n')

dfPrice = pd.read_excel('century.xlsx', sheet_name = 'price')
vol = dfPrice['Volatility'].values[1:]
N = len(vol)
price = dfPrice['Price'].values
dividend = dfPrice['Dividends'].values[1:]
dfEarnings = pd.read_excel('century.xlsx', sheet_name = 'earnings')
earnings = dfEarnings['Earnings'].values
L = 9

def fit(window, inflMode, normMode):
    VolFactors = pd.DataFrame({'const' : 1/vol, 'vol' : 1})
    if inflMode == 'N':
        index = price
        div = dividend
        total = np.array([np.log(index[k+1] + dividend[k]) - np.log(index[k]) for k in range(N)])
        earn = earnings
    if inflMode == 'R':
        cpi = dfEarnings['CPI'].values
        index = cpi[-1]*price/cpi[L:]
        div = cpi[-1]*dividend/cpi[L+1:]
        total = np.array([np.log(index[k+1] + div[k]) - np.log(index[k]) for k in range(N)])
        earn = cpi[-1]*earnings/cpi
    Ntotal = total/vol  
    cumearn = np.array([np.mean(earn[k-window:k]) for k in range(L + 1, L + N + 2)])
    IDY = total - np.diff(np.log(cumearn))
    cumIDY = np.append(np.array([0]), np.cumsum(IDY))
    print('Regression for Valuation Measure')
    if normMode == 'N':
        AllFactors = pd.DataFrame({'const' : 1, 'trend' : range(N), 'Bubble' : -cumIDY[:-1]})
        modelValuation = OLS(IDY, AllFactors).fit()
    if normMode == 'Y':
        AllFactors = pd.DataFrame({'const' : 1/vol, 'trend' : range(N)/vol, 'Bubble' : -cumIDY[:-1]/vol})
        modelValuation = OLS(IDY/vol, AllFactors).fit()
    print('R^2 = ', modelValuation.rsquared)
    print(modelValuation.params)
    print(modelValuation.pvalues)
    Valuation = cumIDY - np.array(range(N+1)) * (modelValuation.params['trend'] / modelValuation.params['Bubble'])
    print('Average Valuation = ', np.mean(Valuation))
    print('Current Valuation = ', Valuation[-1])
    resValuation = modelValuation.resid
    analysis(resValuation, 'bubble')

for window in range(1, 11):
    print('window = ', window)
    fit(window, 'N', 'N')
    fit(window, 'R', 'N')
    fit(window, 'N', 'Y')
    fit(window, 'R', 'Y')
    