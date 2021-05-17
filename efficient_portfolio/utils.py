
import pandas as pd
import copy
import numpy as np
def get_allocation(weights, meanReturns):
    allocation = pd.DataFrame(
        weights, index=copy.deepcopy(meanReturns.index), columns=["allocation"]
    )
    # allocation["allocation"]  = [round(i*100,0) for i in allocation["allocation"]]
    return allocation[allocation["allocation"] >= 0.001]

def get_annual_returns_and_covariances(df, start_index, end_index,drop_na=True,coins=None):
      
    if end_index:
        timeframe = df.iloc[start_index:end_index]
    else:
        timeframe = df.iloc[start_index:]
    if coins:
        for c in timeframe.columns:
            if c not in coins:
                del timeframe[c]
    if drop_na:
        timeframe = timeframe.dropna(axis=1, how="any")
    returns_weekly = timeframe.pct_change()
    returns_annual = returns_weekly.mean() * len(timeframe)
    cov_weekly = returns_weekly.cov()
    cov_annual = cov_weekly * len(timeframe)
    return returns_annual, cov_annual


# risk free rate from germany https://www.statista.com/statistics/885915/average-risk-free-rate-select-countries-europe/
def sharp_ratio(r, v, risk_free_rate=0.008):
    return (r - risk_free_rate) / v


def get_returns_and_volatility(returns_annual, weights, covariance):

    returns = weights @ returns_annual
    volatility = np.sqrt(weights.T @ (covariance @ weights)) * np.sqrt(52)
    return returns, volatility

"""
def get_weightened_allocation(df,coins=[]):
    
    if coins:
        for c in df.columns:
            if c not in coins:
                del df[c]

    max_sharp_allocations, sharps = get_max_sharp_allocations(df,[(-52,None)] )
    weights = [1]
    weighted_allocations = [ m*w for m,w in zip(max_sharp_allocations, weights)]
    summed = pd.concat(weighted_allocations, axis=1).sum(axis=1) 
    normed =   summed / summed.sum()
    below_1_percent = normed[normed >= 0.01] 
    renormed = below_1_percent / below_1_percent.sum()
    return renormed, sharps[0]
def calculate_efficient_frontier_for_timeframe(df, start_index, end_index=None):
    
    
    annual_returns, annual_cov = get_annual_returns_and_covariances(df, start_index, end_index)
    efficient_weights = get_efficient_weights(annual_returns, annual_cov)
    
    max_sharp_allocation = None
    sharp = -np.inf
    for weights in efficient_weights:
        r,v = get_returns_and_volatility(annual_returns, weights, annual_cov)
        allocation = get_allocation(weights,annual_returns)
        if r/v > sharp:
            sharp = r/v
            max_sharp_allocation = allocation
        #print(allocation)
    return sharp, max_sharp_allocation
"""