import sys
import datetime
import re
import csv

with open(sys.argv[1], 'r') as f:
    data=f.readlines()

extra=data[0]
data=data[1:]
title=data[0].split()
csv_list=[]
for i in data:
    print(i)
    i=i.replace('\n', '')
    tmp=re.split(r"[ |]",i)
    x= str(datetime.datetime.now().strftime("%d-%m"))
    if x in tmp:
        tmp.remove(x)
    if '\n' in tmp:
        tmp.remove('\n')
    c = tmp.count('')
    for j in range(c):
        tmp.remove('') 
    csv_list.append(tmp)

for i in csv_list:
    print(i)

with open(sys.argv[2], "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_list)
