"""
Created on Mon Apr 24 21:19:09 2017

@author: Jyo
"""

from bs4 import BeautifulSoup
import requests
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
import os


### This part of the code is used to get links to all mutual funds filings from the SEC EDGAR database ###

# Scrape the SEC EDGAR website for all mutual funds with filings from startYr
def getCIKs(startYr):

    EDGARDBPrefix = "https://www.sec.gov/Archives/"
    CIKs = []

    indexUrlList = []
    yr = startYr
    while yr < dt.datetime.now().year:
        indexUrlList.append(EDGARDBPrefix + 'edgar/full-index/' + str(yr) + '/QTR1/company.idx')
        indexUrlList.append(EDGARDBPrefix + 'edgar/full-index/' + str(yr) + '/QTR2/company.idx')
        indexUrlList.append(EDGARDBPrefix + 'edgar/full-index/' + str(yr) + '/QTR3/company.idx')
        indexUrlList.append(EDGARDBPrefix + 'edgar/full-index/' + str(yr) + '/QTR4/company.idx')
        yr = yr + 1

    for url in indexUrlList:
        print("Accessing... "+url)
        response = requests.get(url)
        for l in response.iter_lines():
            line = str(l)
            if "NSAR-" in line:

                # Add to CIK list if it isn't already present in the list
                CIK = int(line[74:86])
                if CIK not in CIKs:
                    CIKs.append(CIK)

    CIKs = sorted(CIKs)
    print(len(CIKs))
    return CIKs

# Scrape the SEC EDGAR website for all mutual fund filings from startYr
def getFilings(startYr):

    EDGARDBPrefix = "https://www.sec.gov/Archives/"

    indexUrlList = []
    yr = startYr
    while yr < dt.datetime.now().year:
        indexUrlList.append(EDGARDBPrefix + 'edgar/full-index/' + str(yr) + '/QTR1/company.idx')
        indexUrlList.append(EDGARDBPrefix + 'edgar/full-index/' + str(yr) + '/QTR2/company.idx')
        indexUrlList.append(EDGARDBPrefix + 'edgar/full-index/' + str(yr) + '/QTR3/company.idx')
        indexUrlList.append(EDGARDBPrefix + 'edgar/full-index/' + str(yr) + '/QTR4/company.idx')
        yr = yr + 1

    tempList = []
    for url in indexUrlList:
        print("Accessing... "+url)
        response = requests.get(url)
        for l in response.iter_lines():
            line = str(l)
            if "NSAR-" in line:
                CIK = int(line[74:86])
                filingType = line[61:74]
                filingDate = line[86:98]
                EDGARLink = EDGARDBPrefix+line[100:143]

                tempList.append([CIK, filingType, filingDate, EDGARLink])

    cols = ["CIK", "Filing type", "Filing date", "EDGAR link"]
    return pd.DataFrame(tempList, columns=cols)

Filename = "EDGAR Filing links.csv"
refresh = True
startYr = 2008

# Get list of CIKs to process
MFCIKList = getCIKs(startYr)

# Get filing links and save to CSV
MFData = getFilings(startYr)
MFData.to_csv(Filename, index=False)


### This part of the code is used to generate optimal portfolios using bitcoin, equities and the dollar index ###

os.chdir('/Users/Jyo/Desktop/DI/bitcoin')

# Turn off progress printing
solvers.options['show_progress'] = False


def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)


def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    if sigma > 10:
        return random_portfolio(returns)
    return mu, sigma


def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 50
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    # for i in range(N):
    #     print(str(list(portfolios[i])[0])+','+str(list(portfolios[i])[1])+','+str(list(portfolios[i])[2]))

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    return portfolios, returns, risks


bitcoin_prices = pd.read_csv(
    'market-price-offl.csv',
    header=None,
    index_col=0,
    names=['date','price_btc'],
    dtype={'price_btc': np.float64},
    skiprows=296,
)
bitcoin_prices['logret_btc'] = np.log(bitcoin_prices/bitcoin_prices.shift())

djtm_prices = pd.read_csv(
    'DJTM.csv',
    header=0,
    index_col=0,
    usecols=['date', 'price_djtm'],
    dtype={'price_djtm': np.float64},
)
djtm_prices['logret_djtm'] = np.log(djtm_prices/djtm_prices.shift())

dxy_prices = pd.read_csv(
    'DB Powershares USD.csv',
    header=0,
    index_col=0,
    usecols=['date', 'price_dxy'],
    dtype={'price_dxy': np.float64},
)
dxy_prices['logret_dxy'] = np.log(dxy_prices/dxy_prices.shift())

all_prices = djtm_prices.join([bitcoin_prices,dxy_prices])
all_prices = all_prices.fillna('').query("logret_btc != '' and logret_djtm != '' and logret_dxy != ''").iloc[47:]

return_vec = all_prices.as_matrix(columns=['logret_djtm', 'logret_btc', 'logret_dxy']).astype(np.double).T

n_portfolios = 100
means, stds = np.column_stack([
    random_portfolio(return_vec)
    for _ in range(n_portfolios)
])
weights, returns, risks = optimal_portfolio(return_vec)

fig1 = plt.figure(1)
plt.plot(stds, means, 'ys')
plt.plot(risks, returns, 'g-o')
plt.ylabel('Return')
plt.xlabel('Volatility')
plt.text(0.02, .006, r'Efficient Frontier', color='green')
plt.title('Return vs Vol for various portfolios including investment in Bitcoin')

fig2 = plt.figure(2)
portfolios = (
    'Portfolio 1',
    'Portfolio 2',
    'Portfolio 3',
    'Portfolio 4',
    'Portfolio 5'
)
plot_portfolios = 5
y_pos = np.arange(plot_portfolios)
djtm_wts = [
    list(weights[15])[0],
    list(weights[19])[0],
    list(weights[25])[0],
    list(weights[29])[0],
    list(weights[35])[0],
]
btc_wts = [
    list(weights[15])[1],
    list(weights[19])[1],
    list(weights[25])[1],
    list(weights[29])[1],
    list(weights[35])[1],
]
dxy_wts = [
    list(weights[15])[2],
    list(weights[19])[2],
    list(weights[24])[2],
    list(weights[29])[2],
    list(weights[35])[2],
]

plt.bar(y_pos+0.4, btc_wts, 0.2, color='b', label='Bitcoin')
plt.bar(y_pos, djtm_wts, 0.2, color='g', label='DJ Total Market Index')
plt.bar(y_pos+0.2, dxy_wts, 0.2, color='y', label='DB Dollar Index')

plt.xticks(y_pos, portfolios)
plt.ylabel('Weight')
plt.title('Sample Efficient Portfolio Weights')
plt.legend()

plt.show()
