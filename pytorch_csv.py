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

for i in range(len(data)-2):
    if i==31:
        print("31 row:")
        print(data[i].split("  "))
    tmp=data[i].split("  ")
    
    c = tmp.count('')
    for j in range(c):
        tmp.remove('')
    try:
        tmp.remove("\n")
    except:
        pass
    #print(tmp)
    for i in range(len(tmp)):
        match = re.search(r"\d\.\d{3}ms", str(tmp[i].replace(" ","")))
        if match:
            l = tmp[i].replace("ms","").replace(" ","")
            micro=float(l)*1000
            tmp[i]=micro
            continue
        match2 = re.search(r"\d\.\d{3}us", str(tmp[i].replace(" ","")))
        if match2:
            micro=float(tmp[i].replace("us","").replace(" ",""))
            tmp[i]=micro       
    print(tmp)
    csv_list.append(tmp)
csv_list.append([data[-1]])
csv_list.append([data[-2]])

print(tmp)

with open(sys.argv[2], "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_list)
