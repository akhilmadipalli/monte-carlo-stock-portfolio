"""
Implement the Monte Carlo method to simulate a stock portfolio
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from numpy.ma.core import transpose


#import data
def get_data(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end)
    stock_data = stock_data['Close']
    returns = stock_data.pct_change(fill_method=None)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

stock_list = ['AAPL', 'GOOG', 'META', 'MSFT', 'AMZN', 'GME']
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300)

meanReturns, cov_matrix = get_data(stock_list, start_date, end_date)

weights = np.random.rand(len(meanReturns))
weights /= np.sum(weights)

#Monte Carlo Method
#number of simulations
mc_sims = 100
time = 100 #timeframe in days

mean_matrix = np.full(shape=(time, len(weights)), fill_value=meanReturns)
mean_matrix = transpose(mean_matrix)

initialPortfolio = 10000

portfolio_sims = np.full(shape=(time, mc_sims), fill_value=0.0)
for m in range(0, mc_sims):
    # do monte carlo loops
    Z = np.random.normal(size=(time, len(weights)))
    L = np.linalg.cholesky(cov_matrix)
    daily_returns = mean_matrix + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, transpose(daily_returns)) + 1)*initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Porfolio Value ($)')
plt.xlabel('Time (Days)')
plt.title('MC Simul of a Stock Portfolio')
plt.show()



