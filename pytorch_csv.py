import sys
import datetime
import re
import csv

with open(sys.argv[1], 'r') as f:
    data=f.readlines()

data=data[1:]
del data[1]
del data[-1]
del data[-1]
del data[-3]
csv_list=[]
 
for i in range(len(data)):
    tmp=data[i].split("  ")
    print(tmp)
    c = tmp.count('')
    for j in range(c):
        tmp.remove('')
    csv_list.append(tmp)
print(tmp)

with open(sys.argv[2], "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_list)
