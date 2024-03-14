#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""Before we can start really looking for patterns
in stock price time series data, we need to make
sure we have a set of time series that don't correlate
too closely. Otherwise, the neural net (or something)
will likely learn patterns that don't exist because
multiple stocks look the same..."""

from functools import reduce

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from quandl import Quandl

def plot(label, corr, sym1, data1, sym2, data2):
    """
    Just plots. This is helpful to see what either highly
    correlated or non-correlated time series look like.
    """
    assert data1.shape == data2.shape
    time_points = np.arange(data1.shape[0])

    plt.figure(figsize=(10, 6))

    plt.plot(time_points, data1, label=sym1)

    plt.plot(time_points, data2, label=sym2)

    plt.legend()

    plt.title(f"Comparison of {sym1:s} vs {sym2:s} ({label:s}: {corr:.7f})")
    plt.xlabel("Time Point")
    plt.ylabel("Value")

    plt.show()

#pylint: disable=too-many-locals
def main():
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

    return_data = (raw_data[:, 1:] - raw_data[:, 0:-1]) / raw_data[:, 0:-1]
    # normalize
    centered_return_data = return_data - return_data.mean(axis=1, keepdims=True)

    """
    calculate correlations
    if we find highly correlated sets, one of the stocks in the set will be added 
    to the blacklist
    """
    cov = np.matmul(centered_return_data, centered_return_data.T) / (len(common_dates)-2)
    cov2 = np.cov(centered_return_data)
    epsilon = 1e-15
    assert np.linalg.norm(cov2 - cov, 2) < epsilon

    corr = cov / np.matmul(np.sqrt(np.diagonal(cov)).reshape(-1, 1), np.sqrt(np.diagonal(cov)).reshape(1, -1))
    corr_zero_diag = corr.copy()
    corr_zero_diag[np.arange(0, len(symbols)), np.arange(0, len(symbols))] = 0

    """
    find the max/min correlations and view them.
    this gives us an idea of what a highly correlated set looks like,
    and also what a low correlated set looks like.
    we'll also add one stock from the highly correlated pair(s) to the 
    blacklist, until there are no sufficiently highly correlated stock pairs.
    """
    minindex = np.argmin(corr_zero_diag)
    minrow, mincol = minindex // len(symbols), minindex % len(symbols)
    maxindex = np.argmax(corr_zero_diag)
    maxrow, maxcol = maxindex // len(symbols), maxindex % len(symbols)

    plot("Maximum Correlation", corr[maxrow, maxcol], symbols[maxrow], raw_data[maxrow], symbols[maxcol], raw_data[maxcol])
    plot("Minimum Correlation", corr[minrow, mincol], symbols[minrow], raw_data[minrow], symbols[mincol], raw_data[mincol])
#pylint: enable=too-many-locals



if __name__ == "__main__":
    main()
