#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""Before we can start really looking for patterns
in stock price time series data, we need to make
sure we have a set of time series that don't correlate
too closely. Otherwise, the neural net (or something)
will likely learn patterns that don't exist because
multiple stocks look the same..."""

import csv
import argparse

from functools import reduce
from collections import Counter

from tqdm import tqdm
import numpy as np
import cvxopt

from quandl import Quandl
from decorrelate import plot

def compute_covariance_matrix(data):
    # normalize
    centered_data = data - data.mean(axis=1, keepdims=True)
    cov = np.matmul(centered_data, centered_data.T) / (centered_data.shape[1]-1)
    cov2 = np.cov(centered_data)
    epsilon = 1e-15
    assert np.linalg.norm(cov2 - cov, 2) < epsilon

    return cov

#pylint: disable=too-many-locals
def main():
    """This main function is horribly bloated and should be split into smaller functions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-tolerance", "-r", type=float, required=True)
    args = parser.parse_args()
    quandl = Quandl()
    nasdaq100 = [
            "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD",
            "AMGN", "AMZN", "ANSS", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR",
            "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD",
            "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DLTR", "DXCM",
            "EA", "EXC", "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG",
            "GOOGL", "HON", "IDXX", "ILMN", "INTC", "INTU", "ISRG", "KDP", "KHC",
            "KLAC", "LRCX", "LULU", "MAR", "MCHP", "MDB", "MDLZ", "MELI", "META",
            "MNST", "MRNA", "MRVL", "MSFT", "MU", "NFLX", "NVDA", "NXPI", "ODFL",
            "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PYPL", "QCOM",
            "REGN", "ROP", "ROST", "SBUX", "SIRI", "SNPS", "SPLK", "TEAM", "TMUS",
            "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBA", "WBD", "WDAY",
            "XEL", "ZS"
        ]
    # BKNG was removed because it has very few dates in common with everything else.
    #      (After keeping only common dates, all that remained were
    #       ['2018-03-08', '2018-03-09', '2018-03-12', '2018-03-13', '2018-03-14',
    #        '2018-03-15', '2018-03-16', '2018-03-19', '2018-03-20', '2018-03-21',
    #        '2018-03-22', '2018-03-23', '2018-03-26', '2018-03-27'])
    # GOOGL was removed because it is nearly perfectly correlated with GOGL
    # MNST was _originally_ removed because I (Binesh) forgot about stock splits.
    #      So, now it's not being removed, since we're using adjusted closes.
    # ODFL was also removed for some reason, not sure why, so we put it back
    #      in.
    # ("Removed" in the above context, means "added in the blacklist.")
    blacklist = set([
        "BKNG", "GOOGL"
    ])

    all_stocks = {}
    for symbol in tqdm(nasdaq100, ncols=0, leave=True, desc="nasdaq100"):
        tsd = quandl.grab_time_series_daily(symbol)
        if symbol not in blacklist and "quandl_error" not in tsd:
            all_stocks[symbol] = tsd

    all_dates = set(datum[0] for _, full in all_stocks.items() for datum in full["dataset"]["data"])
    common_dates = sorted(reduce(lambda acc, x: acc & x, [set(datum[0] for datum in full["dataset"]["data"]) for sym, full in all_stocks.items()], all_dates))
    print(common_dates)

    symbols = sorted(set(key for key, _ in all_stocks.items()))

    massaged = {symbol: {datum[0]: datum for datum in all_stocks[symbol]["dataset"]["data"]} for symbol in symbols}

    # 4 is _raw_ close
    # 11 is _adjusted_ (for splits) close
    raw_data = np.array([[massaged[symbol][date][11] for date in common_dates] for symbol in symbols])
    # raw_data is of size (len(symbol), len(common_dates))

    return_data = np.log(raw_data[:, 1:]) - np.log(raw_data[:, :-1]) # log(today/yesterday)
    # return_data is of size (len(symbol), len(common_dates)-1)

    # calculate correlations
    # if we find highly correlated sets, one of the stocks in the set will be added
    # to the blacklist

    mean_returns = return_data.mean(axis=1, keepdims=True)
                                                    # In the terms of https://en.wikipedia.org/wiki/Modern_portfolio_theory#Efficient_frontier_with_no_risk-free_asset
                                                    # our "q" is their "R^T" (average returns)

    # (yesterday + today - yesterday) / yesterday
    # r = exp(52 * 5 * log(1 + (today - yesterday) / yesterday))
    yearly_returns = 52*5*mean_returns
    cov = compute_covariance_matrix(return_data)

    # We want the minimum variance portfolio
    # Minimize [p1 p2 p3 .. pn] [cov] Transpose([p1 p2 p3 ... pn])
    # call [p1 ... pn] "x" in the cvxopt docs
    # call [cov] = "P/2" in the cvxopt docs So, P = 2 * cov
    # our "q" in the cvxopt docs is np.zeros(1, num_stocks)
    # subject to:
    #      p1 < 1
    #      -p1 < 0
    #      G = np.concatenate(np.eye(num_stocks), -np.eye(num_stocks))
    #      h = np.concatenate([np.ones(1, num_stocks), np.zeros(1, num_stocks)], axis=0)
    #      [1 1 1 1 1... ] . x == 1 # The sum of all the proportions is 1
    #      A = np.ones(num_stocks, 1)
    #      b = np.ones(1, 1)

    num_stocks = len(symbols)

    # These names correspond to the docs in cvxopt.
    #pylint: disable=invalid-name
    # https://cvxopt.org/userguide/coneprog.html#quadratic-programming
    # minimize (1/2) x^T P x + q^T x
    # In other words minimize the variance of the portfolio (x)
    P = cvxopt.matrix(2 * cov)

    # In the terms of https://en.wikipedia.org/wiki/Modern_portfolio_theory#Efficient_frontier_with_no_risk-free_asset
    # our "q" (in cvxopt land) is their "q * R^T" (risk tolerance * average returns in MPT land)
    risk_tolerance = args.risk_tolerance # risk tolerance is "q" in MPT land.
    q = cvxopt.matrix(-risk_tolerance * mean_returns) # mean_returns is "R^T" in MPT land. We want to _maximize_ the returns, hence the negative sign.

    # G x <= h
    # all the proportions should be 0 < x < 1
    G = cvxopt.matrix(np.concatenate([np.eye(num_stocks), -np.eye(num_stocks)], axis=0))
    h = cvxopt.matrix(np.concatenate([np.ones(num_stocks), np.zeros(num_stocks)], axis=0))

    # A x = b
    # the sum of the proportions should be 1.
    # TODO: Change this so that A x = b should be the dollar value of our portfolio.
    #       I think this means that instead of A being a "one" matrix, np.ones(1, num_stocks)
    #       it should be [price(AAPL), price(CSCO), ...]
    #       and b should be the how much money we have.
    A = cvxopt.matrix(np.ones((1, num_stocks)))
    b = cvxopt.matrix(np.ones((1, 1)))

    res = cvxopt.solvers.qp(P, q, G, h, A, b)
    x = np.array(list(res["x"])).reshape(num_stocks, 1)
    #pylint: enable=invalid-name

    proportions = Counter(dict(zip(symbols, list(res["x"]))))
    symbol2idx = {symbol: idx for idx, symbol in enumerate(symbols)}

    for symbol, proportion in proportions.most_common():
        print(f"{(proportion*100):.2f}% {symbol:s} return={100 * yearly_returns[symbol2idx[symbol], 0]:.2f}%")

    fixed_order = [
        "PEP", "XEL", "SIRI", "ISRG", "COST",
        "CPRT", "FAST", "VRSK", "CMCSA", "GILD",
        "ORLY", "ODFL", "FANG", "ROP", "AAPL",
        "PCAR", "CTAS", "SBUX", "PANW", "KLAC",
        "QCOM", "MRVL", "ROST", "DLTR", "AEP",
        "REGN", "HON", "CHTR", "ANSS", "SNPS",
        "EA", "AMGN", "WBA", "BIIB", "DXCM",
        "FTNT", "CSX", "EXC", "MCHP", "MAR",
        "CSGP", "ILMN", "ADI", "GOOG", "TMUS",
        "PAYX", "CSCO", "PYPL", "CDNS", "INTC",
        "IDXX", "AMZN", "KHC", "CTSH", "MNST",
        "AVGO", "TXN", "VRTX", "AMAT", "MSFT",
        "ADP", "INTU", "ADSK", "TSLA", "TTWO",
        "SPLK", "ADBE", "NFLX", "NVDA", "WDAY",
        "LRCX", "MDLZ", "MU", "AMD",
    ]

    with open("/tmp/reqs.csv", "wt", encoding="utf-8") as csvfp:
        rows = [{"symbol": symbol, "proportion": proportions[symbol], "price": raw_data[symbol2idx[symbol], -1]} for symbol in fixed_order]
        csvdw = csv.DictWriter(csvfp, fieldnames=["symbol", "proportion", "price"], extrasaction="ignore")
        csvdw.writeheader()
        csvdw.writerows(rows)

    expected_return = np.matmul(yearly_returns.T, x)
    expected_risk = np.sqrt(np.matmul(
        x.T,
        np.matmul(
            cov,
            x
        )
    ))
    print(f"Expected Yearly Return: {100*(np.exp(expected_return)[0, 0]-1):.2f}%")
    print(f"Expected Daily Risk: {100*expected_risk[0, 0]:.2f}%")
#pylint: enable=too-many-locals

if __name__ == "__main__":
    main()
