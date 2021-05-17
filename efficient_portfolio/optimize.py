import numpy as np

import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import minimize
import copy
from utils import get_annual_returns_and_covariances, get_allocation, get_returns_and_volatility,sharp_ratio



def negative_return(weights, returns_annual, covariance):
    r, v = get_returns_and_volatility(returns_annual, weights, covariance)
    return -r


def positive_return(weights, returns_annual, covariance):
    r, v = get_returns_and_volatility(returns_annual, weights, covariance)
    return r


def volatility(weights, returns_annual, covariance):

    r, v = get_returns_and_volatility(returns_annual, weights, covariance)
    return v




def negative_sharp_ratio(r, v, risk_free_rate=0.008):
    return -sharp_ratio(r, v, risk_free_rate=risk_free_rate)


# need to put weights in front for optimizer
def negative_sharp_of_portfolio(weights, returns_annual, covariance, risk_free_rate):
    r, v = get_returns_and_volatility(returns_annual, weights, covariance)
    return negative_sharp_ratio(r, v, risk_free_rate=risk_free_rate)



def maxSR(meanReturns, covMatrix, riskFreeRate = 0.008, constraintSet=(0,1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(negative_sharp_of_portfolio, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def maxSR(meanReturns, covMatrix, riskFreeRate = 0.008, constraintSet=(0,1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(negative_sharp_of_portfolio, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def minimizeVariance(meanReturns, covMatrix, riskFreeRate = 0.008, constraintSet=(0,1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(volatility, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def maxReturn(meanReturns, covMatrix, riskFreeRate = 0.008, constraintSet=(0,1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(negative_return, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0,1)):
    """For each returnTarget, we want to optimise the portfolio for min variance"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)

    constraints = ({'type':'eq', 'fun': lambda x: positive_return(x, meanReturns, covMatrix) - returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = minimize(volatility, numAssets*[1./numAssets], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    return effOpt


def get_efficient_weights(meanReturns, covMatrix):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Return Ratio Portfolio
    maxSR_Portfolio = maxReturn(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = get_returns_and_volatility(meanReturns,maxSR_Portfolio['x'], covMatrix)
    #maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]

    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = get_returns_and_volatility(meanReturns,minVol_Portfolio['x'], covMatrix)
    #minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]

    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target))

    return  [ e.x for e in efficientList]

