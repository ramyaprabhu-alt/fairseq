import pandas as pd
import sys

import re

def ms_to_microseconds(ms_string):
    match = re.match(r'([\s]*[\d.]+)ms', ms_string)
    if match:
        milliseconds = float(match.group(1))
        microseconds = milliseconds * 1000
        return f'{microseconds:.3f}'
    elif re.match(r'([\s]*[\d.]+)us', ms_string):
        return ms_string.replace('us', '')
    else:
        return ms_string


x=pd.read_csv(sys.argv[1])
y=pd.read_csv(sys.argv[2])
final=pd.merge(x,y, on=' Name',how="outer",suffixes=(sys.argv[3], sys.argv[4]))
final.to_csv('moe_vs_dense.csv')
