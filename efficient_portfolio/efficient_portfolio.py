import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import copy
from optimize import get_efficient_weights, get_returns_and_volatility,maxSR
from utils import get_allocation,get_annual_returns_and_covariances,get_returns_and_volatility,sharp_ratio

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
    return df, market_cap


def get_max_sharp_allocation(df, start,end=None):
    
    annual_returns, annual_cov = get_annual_returns_and_covariances(df, start, end)
    max_sr_weights = maxSR(annual_returns, annual_cov).x
    r,v = get_returns_and_volatility(annual_returns, max_sr_weights, annual_cov)
    return get_allocation(max_sr_weights, annual_returns), sharp_ratio(r,v)


def get_weekly_allocations(df, coins):
    date_and_allocation = []
    if coins:
        for c in df.columns:
            if c not in coins:
                del df[c]

    for last_n_weeks in range(len(df),len(df)-1,-1):
        cut_df = df.iloc[:last_n_weeks-1]
        date = df.index[last_n_weeks-1]
        print("Allocation timestamp: ", date)
        allocation, sharp = get_max_sharp_allocation(cut_df, -52, None )
        print("Sharp: ",sharp)
        print(allocation)
        date_and_allocation.append((date,allocation))

    return date_and_allocation

def get_actual_returns(df, allocation):
    pct_changes = df.iloc[-52:].pct_change().iloc[1:]

    for c in  pct_changes.columns:
        if c not in allocation.index: 
            del pct_changes[c]
    return allocation.T @ pct_changes.T

def plot_returns(all_returns):
    start = 100
    values = (all_returns +1).cumprod(axis=1)
    plt.plot(values.T)
    plt.show()
    
if __name__=="__main__":

    df, market_cap = load_data()
    biggest_by_market_cap = list(market_cap.iloc[-1].dropna().sort_values(ascending=False)[:40].index)
    date_and_allocations = get_weekly_allocations(df, biggest_by_market_cap)
    all_returns = get_actual_returns(df, date_and_allocations[0][1])
    plot_returns(all_returns)
    