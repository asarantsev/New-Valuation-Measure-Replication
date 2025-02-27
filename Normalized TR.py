import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from statsmodels.api import OLS

N = 96 # total number of years
W = 5 # window for averaging earnings

annualDF = pd.read_excel('replicatedata.xlsx', sheet_name = 'data')
annual = annualDF.values
div = annual[:N, 1].astype(float) #annual dividends
earn = annual[:N, 2].astype(float) #annual earnings
index = annual[:N + 1, 3].astype(float) #annual index values
cpi = annual[:N + 1, 4].astype(float) #annual consumer price index
TR = annual[:N, 6].astype(float) #annual normalized total real returns

rdiv = cpi[-1]*div/cpi[1:]
rearn = cpi[-1]*earn/cpi[1:]
rindex = cpi[-1]*index/cpi
rwealth = np.append(np.array([1]), np.exp(np.cumsum(TR))) # real wealth
cumearn = [sum(rearn[k:k+W])/W for k in range(N-W+1)] 
growth = np.diff(np.log(cumearn))

IDY = TR[W:] - growth
cumIDY = np.append(np.array([0]), np.cumsum(IDY))

TRearn = rearn*(rwealth[1:]/rindex[1:])
TRcumearn = [sum(TRearn[k:k+W])/W for k in range(N-W+1)]
TRCAPE = rwealth[W:]/TRcumearn

DF = pd.DataFrame({'const' : 1, 'trend' : range(N-W), 'Bubble' : cumIDY[:-1]})
Regression = OLS(IDY, DF).fit()
print(Regression.summary())
coefficients = Regression.params
intercept = coefficients['const']
trend_coeff = coefficients['trend']
bubble_coeff = coefficients['Bubble']

b = 1-bubble_coeff
c=trend_coeff/bubble_coeff
h=(c-intercept)/bubble_coeff
print(b)
print(c)
print(h)

residuals = IDY - Regression.predict(DF)
stderr = np.std(residuals)

aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('Normalized Residuals \n Quantile Quantile Plot vs Normal')
plt.show()
plot_acf(residuals)
plt.title('Normalized Residuals \n ACF for Original Values')
plt.show()
plot_acf(aresiduals)
plt.title('Normalized Residuals \n ACF for Absolute Values')
plt.show()