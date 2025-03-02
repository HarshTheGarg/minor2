#!../.venv/bin/python
import pandas as pd
import numpy as np

def knnPredict(data, pathToCSV, k=5):
  df = pd.read_csv(pathToCSV, header=None)
  df = df.drop_duplicates()

  n = df.shape[0]

  y = df[0]
  x = df.iloc[:, 1:]

  dist, weig = [], []

  for i in range(n):
    d = np.linalg.norm(x.iloc[i]-data)
    dist.append(d)
    if d == 0:
      weig.append(1000)
    else:
      weig.append(1/(d**2))

  sortInd = np.argsort(dist)
  keys = y.unique()
  sums = {i:0 for i in keys}

  for i in range(k):
    si = sortInd[i]
    sums[y.iloc[si]] += weig[si]

  maxSum = max(sums.values())
  cla = [key for key in sums if sums[key] == maxSum][0]
  # print(cla)
  return cla

if __name__ == '__main__':
  data = 0.0,0.0,-0.23636363636363636,0.04242424242424243,-0.48484848484848486,0.024242424242424242,-0.6727272727272727,0.03636363636363636,-0.8181818181818182,0.07878787878787878,-0.5636363636363636,-0.3212121212121212,-0.7696969696969697,-0.4303030303030303,-0.896969696969697,-0.4909090909090909,-1.0,-0.5515151515151515,-0.47878787878787876,-0.4303030303030303,-0.6787878787878788,-0.593939393939394,-0.806060606060606,-0.696969696969697,-0.9090909090909091,-0.7696969696969697,-0.3575757575757576,-0.49696969696969695,-0.503030303030303,-0.703030303030303,-0.6060606060606061,-0.8181818181818182,-0.696969696969697,-0.9090909090909091,-0.2,-0.5151515151515151,-0.23636363636363636,-0.7212121212121212,-0.2787878787878788,-0.8606060606060606,-0.3212121212121212,-0.9757575757575757
  knnPredict(data, "./dataset.csv")
