import pandas as pd
import numpy as np
from pprint import pprint

TEST = False

if TEST:
    alldata = pd.read_csv("../data/lastfmi/all_data.csv",header=0, index_col=0)
else:
    alldata = np.loadtxt("../data/lastfmi/alltrain.txt")
    alldata = pd.DataFrame(alldata)

# print(alldata.head())
# print(alldata.iloc[:10,0])
data = list(alldata.groupby(alldata.iloc[:,0]))
data = list(map(lambda x: (x[0], x[1].to_numpy()[:,1]), data))
item_count = alldata.iloc[:, 1].value_counts()


def split(data, train_pro = 0.9):
    length = float(len(data))
    another = []
    train = []
    for u, items in data:
        np.random.shuffle(items)
        dataout = round(len(items)*(1-train_pro))
        if (len(items) - dataout) <= 2:
            train.append((u, items))
            continue
        i = 0
        outlist = np.zeros(len(items)).astype(np.bool)
        while True:
            if sum(outlist) == dataout or i == len(items):
                break
            if item_count[items[i]] <= 2:
                i += 1
                continue
            else:
                item_count[items[i]] = item_count[items[i]] - 1
                outlist[i] = True
                i += 1
        train.append((u, items[~outlist]))
        another.append((u, items[outlist]))
    return train, another

def data2array(data):
    array = []
    for i in range(len(data)):
        u = data[i][0]
        for item in data[i][1]:
            array.append([u, item])
    return np.array(array).astype(np.int)

if __name__ == "__main__":
    train, vad = split(data)
    print("==================")
    
    train = data2array(train)
    vad = data2array(vad)
    if TEST:
        np.savetxt('../data/lastfmi/test.txt', vad, fmt="%d")
        np.savetxt('../data/lastfmi/alltrain.txt', train, fmt="%d")
        print("test, alltrain")
    else:
        np.savetxt('../data/lastfmi/validation.txt', vad, fmt="%d")
        np.savetxt('../data/lastfmi/train.txt', train, fmt="%d")
        print("validation, train")
    print(len(train))
    print(len(vad))
    print(len(alldata))
    print("==================")
