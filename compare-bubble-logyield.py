import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from statsmodels.api import OLS

W = 5 # window for averaging earnings
L = 10 # lag for earnings
print('Window for averaged earnings = ', W)
data = pd.read_excel('century.xlsx', sheet_name = 'price')
vol = data['Volatility'].values[1:]
div = data['Dividends'].values[1:] #annual dividends
index = data['Price'].values #annual index values
N = len(vol)
data0 = pd.read_excel('century.xlsx', sheet_name = 'earnings')
cpi = data0['CPI'].values #annual consumer price index
earn = data0['Earnings'].values #annual earnings
TR = [np.log(index[k+1] + div[k]) - np.log(index[k]) for k in range(N)]

rdiv = cpi[-1]*div/cpi[L:]
rearn = cpi[-1]*earn/cpi
rindex = cpi[-1]*index/cpi[L-1:]
rTR = [np.log(rindex[k+1] + rdiv[k]) - np.log(rindex[k]) for k in range(N)]

wealth = np.append(np.array([1]), np.exp(np.cumsum(TR))) # nominal wealth
rwealth = np.append(np.array([1]), np.exp(np.cumsum(rTR))) # real wealth
cumrearn = [sum(rearn[k:k+W])/W for k in range(N+1)] # cumulative earnings 
rgrowth = np.diff(np.log(cumrearn))
IDY = rTR - rgrowth
cumIDY = np.append(np.array([0]), np.cumsum(IDY))

DF = pd.DataFrame({'const' : 1/vol, 'trend' : -np.array(range(N))/vol, 'Bubble' : cumIDY[:-1]/vol})
Regression = OLS(IDY/vol, DF).fit()
print('Regression for the new valuation measure')
print(Regression.summary())
coefficients = Regression.params
intercept = coefficients['const']
trend_coeff = coefficients['trend']
bubble_coeff = coefficients['Bubble']
c = trend_coeff / bubble_coeff
print('Long-term difference between returns and growth = ', c)

Valuation = cumIDY - c * np.array(range(N+1))
plt.plot(range(1927, 2025), Valuation)
plt.title('New Valuation Measure')
plt.savefig('valuation.png', dpi=300, bbox_inches='tight')
plt.close()

earnyield = cumrearn/rindex
plt.plot(range(1927, 2025), earnyield)
plt.title('Log Cyclically Adjusted Earnings Yield Window ' + str(W))
plt.savefig('log-yield.png', dpi=300, bbox_inches='tight')
plt.close()

factorsDF = pd.DataFrame({'const' : 1/vol, 'Measure' : Valuation[:-1]/vol, 'Yield' : np.log(earnyield[:-1])/vol, 'Vol' : 1})
Reg = OLS(rTR/vol, factorsDF).fit()
print('\n Full Regression \n')
print(Reg.summary())
residuals = Reg.resid
stderr = np.std(residuals)
print('Analysis of residuals')
print('Standard deviation = ', stderr)
print('Shapiro-Wilk and Jarque-Bera p = ', stats.shapiro(residuals)[1], stats.jarque_bera(residuals)[1])
aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('Full Regression Residuals \n Quantile Quantile Plot vs Normal')
plt.savefig("full-qqplot.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(residuals)
plt.title('Full Regression Residuals \n ACF for Original Values')
plt.savefig("full-acf.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(aresiduals)
plt.title('Full Regression Residuals \n ACF for Absolute Values')
plt.savefig("full-aacf.png", dpi=300, bbox_inches='tight')
plt.close()

bubbleDF = pd.DataFrame({'const' : 1/vol, 'Measure' : Valuation[:-1]/vol, 'Vol' : 1})
Reg = OLS(rTR/vol, bubbleDF).fit()
print('\n Bubble Regression\n')
print(Reg.summary())
residuals = Reg.resid
stderr = np.std(residuals)
print('Analysis of residuals')
print('Standard deviation = ', stderr)
print('Shapiro-Wilk and Jarque-Bera p = ', stats.shapiro(residuals)[1], stats.jarque_bera(residuals)[1])
aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('Bubble Regression Residuals \n Quantile Quantile Plot vs Normal')
plt.savefig("bubble-qqplot.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(residuals)
plt.title('Bubble Regression Residuals \n ACF for Original Values')
plt.savefig("bubble-acf.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(aresiduals)
plt.title('Bubble Regression Residuals \n ACF for Absolute Values')
plt.savefig("bubble-aacf.png", dpi=300, bbox_inches='tight')
plt.close()

yieldDF = pd.DataFrame({'const' : 1/vol, 'Yield' : np.log(earnyield[:-1])/vol, 'Vol' : 1})
Reg = OLS(rTR/vol, yieldDF).fit()
print('\n Yield Regression \n')
print(Reg.summary())
residuals = Reg.resid
stderr = np.std(residuals)
print('Analysis of residuals')
print('Standard deviation = ', stderr)
print('Shapiro-Wilk and Jarque-Bera p = ', stats.shapiro(residuals)[1], stats.jarque_bera(residuals)[1])
aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('Yield Regression Residuals \n Quantile Quantile Plot vs Normal')
plt.savefig("yield-qqplot.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(residuals)
plt.title('Yield Regression Residuals \n ACF for Original Values')
plt.savefig("yield-acf.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(aresiduals)
plt.title('Yield Regression Residuals \n ACF for Absolute Values')
plt.savefig("yield-aacf.png", dpi=300, bbox_inches='tight')
plt.close()

cutDF = pd.DataFrame({'const' : 1/vol, 'Vol' : 1})
Reg = OLS(rTR/vol, cutDF).fit()
print('\n Yield Regression \n')
print(Reg.summary())
residuals = Reg.resid
stderr = np.std(residuals)
print('Analysis of residuals')
print('Standard deviation = ', stderr)
print('Shapiro-Wilk and Jarque-Bera p = ', stats.shapiro(residuals)[1], stats.jarque_bera(residuals)[1])
aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('Cut Regression Residuals \n Quantile Quantile Plot vs Normal')
plt.savefig("cut-qqplot.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(residuals)
plt.title('Cut Regression Residuals \n ACF for Original Values')
plt.savefig("cut-acf.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(aresiduals)
plt.title('Cut Regression Residuals \n ACF for Absolute Values')
plt.savefig("cut-aacf.png", dpi=300, bbox_inches='tight')
plt.close()