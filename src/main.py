#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""Do something with stock data."""

from tqdm import tqdm
from quandl import Quandl

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

    for symbol in tqdm(nasdaq100, ncols=0, leave=True, desc="nasdaq100"):
        dummy_tsd = quandl.grab_time_series_daily(symbol)

if __name__ == "__main__":
    main()
