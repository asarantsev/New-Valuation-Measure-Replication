import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

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
    
def gaussNoise(data, size):
    covMatrix = data.cov()
    simNoise = []
    dim = len(data.keys())
    for sim in range(NSIMS):
        simNoise.append(np.transpose(np.random.multivariate_normal(np.zeros(dim), covMatrix, size)))
    return simNoise

def KDE(data, size):
    method = stats.gaussian_kde(np.transpose(data.values), bw_method = 'silverman')
    simNoise = []
    for sim in range(NSIMS):
        simNoise.append(np.array(method.resample(size)))
    return simNoise

NSIMS = 1000
NDISPLAYS = 5
np.random.seed(0)
dfPrice = pd.read_excel('century.xlsx', sheet_name = 'price')
vol = dfPrice['Volatility'].values[1:]
price = dfPrice['Price'].values
dividend = dfPrice['Dividends'].values
dfEarnings = pd.read_excel('century.xlsx', sheet_name = 'earnings')
earnings = dfEarnings['Earnings'].values
lvol = np.log(vol)
N = len(vol)
RegVol = stats.linregress(lvol[:-1], lvol[1:])
betaVol = RegVol.slope
alphaVol = RegVol.intercept
print('Autoregression Slope = ', betaVol)
print('Autoregression Intercept = ', alphaVol)
resVol = [lvol[k+1] - betaVol * lvol[k] - alphaVol for k in range(N-1)]
meanVol = np.mean(vol)
L = 9

def fit(window, inflMode):
    VolFactors = pd.DataFrame({'const' : 1/vol, 'vol' : 1})
    if inflMode == 'Nominal':
        index = price
        div = dividend
        total = np.array([np.log(index[k+1] + dividend[k+1]) - np.log(index[k]) for k in range(N)])
        earn = earnings
    if inflMode == 'Real':
        cpi = dfEarnings['CPI'].values
        index = cpi[-1]*price/cpi[L:]
        div = cpi[-1]*dividend/cpi[L:]
        total = np.array([np.log(index[k+1] + div[k+1]) - np.log(index[k]) for k in range(N)])
        earn = cpi[-1]*earnings/cpi
    Ntotal = total/vol  
    cumearn = np.array([np.mean(earn[k-window:k]) for k in range(L + 1, L + N + 2)])
    IDY = total - np.diff(np.log(cumearn))
    cumIDY = np.append(np.array([0]), np.cumsum(IDY))
    earngr = np.diff(np.log(earn[L:]))
    growth = earngr/vol
    modelGrowth = OLS(growth, VolFactors).fit()
    print('Regression for Annual Earnings Growth')
    print(modelGrowth.summary())
    resGrowth = modelGrowth.resid
    print('Regression for Total Returns')
    AllFactors = pd.DataFrame({'const' : 1/vol, 'trend' : range(N)/vol, 'Bubble' : -cumIDY[:-1]/vol})
    modelReturns = OLS(IDY/vol, AllFactors).fit()
    print(modelReturns.summary())
    print('Regression for Total Returns including Volatility')
    ExtendedFactors = pd.DataFrame({'const' : 1/vol, 'trend' : range(N)/vol, 'Bubble' : -cumIDY[:-1]/vol, 'vol' : 1})
    extended = OLS(IDY/vol, ExtendedFactors).fit()
    print(extended.summary())
    Valuation = cumIDY - np.array(range(N+1)) * (modelReturns.params['trend'] / modelReturns.params['Bubble'])
    plt.plot(range(1928, 1929 + N), Valuation)
    plt.title('New Valuation Measure')
    plt.savefig('NewMeasure.png')
    plt.close()
    resReturns = modelReturns.resid
    allResids = pd.DataFrame({'Volatility' : resVol, 'Growth': resGrowth[1:], 'Returns' : resReturns[1:]})
    allMeans = {'Volatility' : round(meanVol, 1), 'Measure' : round(np.mean(Valuation), 3)}
    allCurrents = {'Volatility' : round(vol[-1], 1), 'Earnings' : earn[-window:], 'Measure' : round(Valuation[-1], 3)}
    allModels = {'Growth' : modelGrowth, 'Returns' : modelReturns}
    for resid in allResids:
        plots(allResids[resid], resid)
    return modelGrowth, modelReturns, allResids, allMeans, allCurrents
   
def simReturns(window, inflMode, residMode, horizon):
    modelGrowth, modelReturns, allResids, allMeans, allCurrents = fit(window, inflMode)
    if residMode == 'Gauss':
        innovations = gaussNoise(allResids, horizon)
    if residMode == 'KDE':
        innovations = KDE(allResids, horizon)
    allSims = [] 
    for sim in range(NSIMS):
        simLVol = [np.log(allCurrents['Volatility'])]
        innovation = innovations[sim]
        for t in range(horizon):
            simLVol.append(simLVol[-1]*betaVol + alphaVol + innovation[0, t])
        simVol = np.exp(simLVol)
        simGrowth = [simVol[t+1] * (modelGrowth.predict([1/simVol[t+1], 1])[0] + innovation[1, t]) for t in range(horizon)]
        oldEarn = allCurrents['Earnings']
        simEarn = np.append(oldEarn, oldEarn[-1] * np.exp(np.cumsum(simGrowth)))
        cumSimEarn = np.array([np.mean(simEarn[t:t + window]) for t in range(horizon+1)])
        simMeasure = [allCurrents['Measure']]
        slope = 1 - modelReturns.params['Bubble']
        trend = modelReturns.params['trend']/modelReturns.params['Bubble']
        intercept = modelReturns.params['const'] - trend 
        for t in range(horizon):
            simMeasure.append(simMeasure[t] * slope + intercept + simVol[t+1] * innovation[2, t])
        current = {'Volatility' : round(allCurrents['Volatility'], 1), 'Measure' : round(allCurrents['Measure'], 3)}
        simRet = np.diff(simMeasure) + np.ones(horizon) * trend + np.diff(np.log(cumSimEarn))
        allSims.append(simRet)
    return np.array(allSims), current, allMeans

def output(window, inflMode, residMode, horizon, initialWealth, flow):
    if flow == 0:
        flowText = 'No regular contributions or withdrawals'
    if flow > 0:
        flowText = 'Contributions ' + str(flow) + ' per year'
    if flow < 0:
        flowText = 'Withdrawals ' + str(abs(flow)) + ' per year'
    paths = []
    timeAvgRets = []
    simulatedReturns, current, means = simReturns(window, inflMode, residMode, horizon)
    for sim in range(NSIMS):
        path = [initialWealth]
        simReturn = simulatedReturns[sim]
        timeAvgRets.append(np.mean(simReturn))
        for t in range(horizon):
            if (path[t] == 0):
                path.append(0)
            else:
                new = max(path[t] * np.exp(simReturn[t]) + flow, 0)
                path.append(new)
        paths.append(path)
    paths = np.array(paths)
    avgRet = np.mean([timeAvgRets[sim] for sim in range(NSIMS) if paths[sim, -1] > 0])
    wealthMean = np.mean(paths[:, -1])
    meanProb = np.mean([paths[sim, -1] > wealthMean for sim in range(NSIMS)])
    ruinProb = np.mean([paths[sim, -1] == 0 for sim in range(NSIMS)])
    sortedIndices = np.argsort(paths[:, -1])
    selectedIndices = [sortedIndices[int(NSIMS*(2*k+1)/(2*NDISPLAYS))] for k in range(NDISPLAYS)]
    times = range(horizon + 1)
    simText = str(NSIMS) + ' of simulations'
    timeHorizonText = 'Time Horizon: ' + str(horizon) + ' years'
    inflText = inflMode + ' returns'
    initWealthText = 'Initial Wealth ' + str(round(initialWealth))
    Portfolio = 'The portfolio: 100% Large Stocks'
    modelText = 'Modeling using volatility and the new valuation measure\n with trailing earnings averaged over the past ' + str(window) + ' Y'
    initMarketText = 'Initial (Current) conditions: ' + ' '.join([key + ' ' + str(current[key]) for key in ['Volatility', 'Measure']])
    avgMarketText = 'Historical averages: ' + ' '.join([key + ' ' + str(means[key]) for key in ['Volatility', 'Measure']])
    SetupText = 'SETUP: ' + simText + '\n' + modelText + '\n' + Portfolio + '\n' + timeHorizonText + '\n' + inflText + '\n' + initWealthText +'\n' + initMarketText + '\n' + avgMarketText + '\n' + flowText + '\n'
    if np.isnan(avgRet):
        ResultText = 'RESULTS: 100% Ruin Probability, always zero wealth'
    else:
        RuinProbText = str(round(100*ruinProb, 2)) + '% Ruin Probability'
        AvgRetText = 'time averaged annual returns:\naverage over all paths without ruin ' + str(round(100*avgRet, 2)) + '%'
        MeanText = 'average final wealth ' + str(round(wealthMean))
        MeanCompText = 'final wealth exceeds average with probability ' + str(round(100*meanProb, 2)) + '%'
        ResultText = 'RESULTS: ' + RuinProbText + '\n' + AvgRetText + '\n' + MeanText + '\n' + MeanCompText
    bigTitle = SetupText + '\n' + ResultText + '\n'
    plt.plot([0, 4000], color = 'w', label = bigTitle)
    for display in range(NDISPLAYS):
        index = selectedIndices[display]
        rankText = ' final wealth, ranked ' + str(round(100*(2*display + 1)/(2*NDISPLAYS))) + '% '
        selectTerminalWealth = round(paths[index, -1])
        if (selectTerminalWealth == 0):
            plt.plot(times, paths[index], label = '0' + rankText + 'Gone Bust !!!')
        else:
            plt.plot(times, paths[index], label = str(selectTerminalWealth) + rankText + 'returns: ' + str(round(100*timeAvgRets[index], 2)) + '%')
    plt.xlabel('Years')
    plt.ylabel('Wealth')
    plt.title('Wealth Plot')
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', prop={'size': 11})
    image_path = 'wealth.png'
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

output(7, 'Real', 'KDE', 30, 1000, -40)