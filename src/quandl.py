#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This is going to grab stock price info (as time series)
from quandl. We're going to use this info find patterns
with the time series."""

import os
import re
import json
import time

import requests

def localfile(fname):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), fname))

#pylint: disable=too-few-public-methods
class Quandl:
    def __init__(self):
        with open(localfile("../secrets/quandl.json"), "rt", encoding="utf-8-sig") as jsfp:
            config = json.load(jsfp)
            self.api_key = config["key"]

    def grab_time_series_daily(self, symbol):
        cachefn = localfile(f"cache/quandl/{symbol:s}.json")
        if (not os.path.exists(cachefn)) or (os.stat(cachefn).st_mtime < time.time() - 86400*365):
            os.makedirs(os.path.dirname(cachefn), mode=0o775, exist_ok=True)
            dataset = "WIKI"
            url = f"https://www.quandl.com/api/v3/datasets/{dataset:s}/{symbol:s}.json?api_key={self.api_key:s}"
            resp = requests.get(url, timeout=10)
            tmpfn = re.sub(r'\.json$', '-tmp.json', cachefn)
            with open(tmpfn, "wt", encoding="utf-8") as jsfp:
                json.dump(resp.json(), jsfp, indent=4, sort_keys=True)
            os.rename(tmpfn, cachefn)

        with open(cachefn, "rt", encoding="utf-8-sig") as jsfp:
            return json.load(jsfp)
#pylint: enable=too-few-public-methods
