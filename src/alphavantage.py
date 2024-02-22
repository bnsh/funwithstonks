#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This is the interface to alphavantage."""

import os
import re
import json
import time

import requests

def localfile(fname):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), fname))

#pylint: disable=too-few-public-methods
class AlphaVantage:
    def __init__(self):
        with open(localfile("../secrets/alphavantage.json"), "rt", encoding="utf-8-sig") as jsfp:
            config = json.load(jsfp)
            self.api_key = config["key"]

    def grab_time_series_daily(self, symbol):
        cachefn = localfile(f"cache/alphavantage/{symbol:s}.json")
        if (not os.path.exists(cachefn)) or (os.stat(cachefn).st_mtime < time.time() - 21600):
            os.makedirs(os.path.dirname(cachefn), mode=0o775, exist_ok=True)
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol:s}&apikey={self.api_key:s}&datatype=json"
            resp = requests.get(url, timeout=10)
            tmpfn = re.sub(r'\.json$', '-tmp.json', cachefn)
            with open(tmpfn, "wt", encoding="utf-8") as jsfp:
                json.dump(resp.json(), jsfp, indent=4, sort_keys=True)
            os.rename(tmpfn, cachefn)

        with open(cachefn, "rt", encoding="utf-8-sig") as jsfp:
            return json.load(jsfp)
#pylint: enable=too-few-public-methods
