import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from statsmodels.api import OLS

W = 10 # window for averaging earnings
print('Window for averaged earnings = ', W)
data = pd.read_excel('century.xlsx', sheet_name = 'price')
volatility = data['Volatility'].values[1:]
div = data['Dividends'].values[1:] #annual dividends
index = data['Price'].values #annual index values
N = len(volatility)
data0 = pd.read_excel('century.xlsx', sheet_name = 'earnings')
cpi = data0['CPI'].values #annual consumer price index
earn = data0['Earnings'].values #annual earnings
TR = [np.log(index[k+1] + div[k]) - np.log(index[k]) for k in range(N)]
rdiv = cpi[-1]*div/cpi[W:]
rearn = cpi[-1]*earn/cpi
rindex = cpi[-1]*index/cpi[W-1:]
rTR = [np.log(rindex[k+1] + rdiv[k]) - np.log(rindex[k]) for k in range(N)]
wealth = np.append(np.array([1]), np.exp(np.cumsum(TR))) # nominal wealth
rwealth = np.append(np.array([1]), np.exp(np.cumsum(rTR))) # real wealth
cumrearn = [sum(rearn[k:k+W])/W for k in range(N+1)] # cumulative earnings 
rgrowth = np.diff(np.log(cumrearn))
IDY = (rTR - rgrowth)/volatility
cumIDY = np.append(np.array([0]), np.cumsum(IDY))
print('Normalize implied dividend yield before regression')
DF = pd.DataFrame({'const' : 1, 'trend' : -np.array(range(N)), 'Bubble' : cumIDY[:-1]})
Regression = OLS(IDY, DF).fit()
print('Regression for the new valuation measure')
print(Regression.summary())
coefficients = Regression.params
intercept = coefficients['const']
trend_coeff = coefficients['trend']
bubble_coeff = coefficients['Bubble']
c = trend_coeff / bubble_coeff
print('Long-term difference between returns and growth = ', c)
NewValuation = cumIDY - c * np.array(range(N+1))
plt.plot(range(1927, 2025), NewValuation)
plt.title('Normalized Valuation Measure')
plt.savefig('new-valuation.png', dpi=300, bbox_inches='tight')
residuals = IDY - Regression.predict(DF)
stderr = np.std(residuals)
print('Analysis of residuals')
print('Standard deviation = ', stderr)
print('Shapiro-Wilk and Jarque-Bera p = ', stats.shapiro(residuals)[1], stats.jarque_bera(residuals)[1])
aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('Regression Residuals \n Quantile Quantile Plot vs Normal')
plt.savefig("new-qqplot.png", dpi=300, bbox_inches='tight')
plot_acf(residuals)
plt.title('Regression Residuals \n ACF for Original Values')
plt.savefig("new-acf.png", dpi=300, bbox_inches='tight')
plot_acf(aresiduals)
plt.title('New Regression Residuals \n ACF for Absolute Values')
plt.savefig("new-aacf.png", dpi=300, bbox_inches='tight')