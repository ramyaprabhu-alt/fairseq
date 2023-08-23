import pandas as pd
import sys

import re

x=pd.read_csv(sys.argv[1])
y=pd.read_csv(sys.argv[2])
final=pd.merge(x,y, on=' Name',how="outer",suffixes=(sys.argv[3], sys.argv[4]))
final.to_csv('moe_vs_dense.csv')
