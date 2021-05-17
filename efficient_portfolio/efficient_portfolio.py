import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize
import copy


DELETE_LIST = [
    "UNUS SED LEO",
    "Revain",
    "Celsius",
    "Nexo",
    "Bitcoin Diamond",
    "FTX Token",
    "Ethereum Classic",
    "Crypto.com Coin",
]


def load_data():

    all_data = json.load(open("/home/cehmann/workspaces/stonks/data/crypto/data.json"))
    extracted = []
    is_stable = {}
    mc = []
    for date, data in all_data.items():
        weekly = {"date": date}
        m = {"date": date}
        for coin in data:
            weekly[coin["name"]] = coin["quote"]["USD"]["price"]
            is_stable[coin["name"]] = "stablecoin" in coin["tags"]
            m[coin["name"]] = coin["quote"]["USD"]["market_cap"]
        extracted.append(weekly)
        mc.append(m)
    df = pd.DataFrame.from_records(extracted)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    market_cap = pd.DataFrame.from_records(mc)
    market_cap["date"] = pd.to_datetime(market_cap["date"])
    market_cap = market_cap.set_index("date")

    for c in df.columns:
        if is_stable[c] or c in DELETE_LIST:
            del df[c]
            del market_cap[c]


def get_coins_by_market_cap(market_cap):
    return market_cap.iloc[-1].dropna().sort_values(ascending=False)[:40].index


def get_returns_and_volatility(returns_annual, weights, covariance):

    returns = weights @ returns_annual
    volatility = np.sqrt(weights.T @ (covariance @ weights)) * np.sqrt(52)
    return returns, volatility


def get_allocation(weights, meanReturns):
    allocation = pd.DataFrame(
        weights, index=copy.deepcopy(meanReturns.index), columns=["allocation"]
    )
    # allocation["allocation"]  = [round(i*100,0) for i in allocation["allocation"]]
    return allocation[allocation["allocation"] >= 0.001]


def get_random_weights(n):

    weights = np.random.random(n)
    weights /= np.sum(weights)
    return weights


def negative_return(weights, returns_annual, covariance):
    r, v = get_returns_and_volatility(returns_annual, weights, covariance)
    return -r


def positive_return(weights, returns_annual, covariance):
    r, v = get_returns_and_volatility(returns_annual, weights, covariance)
    return r


def volatility(weights, returns_annual, covariance):

    r, v = get_returns_and_volatility(returns_annual, weights, covariance)
    return v


# risk free rate from germany https://www.statista.com/statistics/885915/average-risk-free-rate-select-countries-europe/
def sharp_ratio(r, v, risk_free_rate=0.008):
    return (r - risk_free_rate) / v


def negative_sharp_ratio(r, v, risk_free_rate=0.008):
    return -sharp_ratio(r, v, risk_free_rate=risk_free_rate)


# need to put weights in front for optimizer
def negative_sharp_of_portfolio(weights, returns_annual, covariance, risk_free_rate):
    r, v = get_returns_and_volatility(returns_annual, weights, covariance)
    return negative_sharp_ratio(r, v, risk_free_rate=risk_free_rate)




def get_actual_returns(df, allocations):
    pct_changes = df.pct_change()
    all_returns = []
    for date, a in allocations:
        real_changes = copy.deepcopy(pct_changes.loc[date])
        #print(real_changes)
        #print(real_changes[a.index])
        pct_returns = pd.concat([a,real_changes],axis=1).dropna().cumprod(axis=1).sum().iloc[-1]
        all_returns.append(pct_returns)
    return all_returns



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
    return returns_annual,cov_annual

def calculate_best_portfolio_for_timeframe(df, start_index, end_index=None):
    
    annual_returns, annual_cov = get_annual_returns_and_covariances(df, start_index, end_index)
    weights = maxSR(annual_returns, annual_cov).x
    r,v = get_returns_and_volatility(annual_returns, weights, annual_cov)
    allocation = get_allocation(weights,annual_returns)
    print(allocation)

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



def get_max_sharp_allocations(df, start_and_end):
    
    max_sharp_allocations = []
    sharps = []
    for start,end in start_and_end:
        print(start,end)
        sharp,allocation = calculate_efficient_frontier_for_timeframe(df, start,end_index=end)
        max_sharp_allocations.append(allocation)
        sharps.append(sharp)
    return max_sharp_allocations, sharps


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



if __name__=="__main__":


all_returns = get_actual_returns(df, date_and_prediction)