#!../.venv/bin/python
import numpy as np
import pandas as pd

import sys
np.set_printoptions(threshold=sys.maxsize)

def kmeansModel(pathToCSV):
  df = pd.read_csv(pathToCSV, header=None)
  df.drop_duplicates()


  iterCount = 20
  k = 3

  centroids = []
  clusterPoints = []

  centroids.append(df.iloc[0, 1:])
  centroids.append(df.iloc[1295, 1:])
  centroids.append(df.iloc[3593, 1:])
  
  def initiateClusterPoints():
    clusterPoints = []
    for _ in range(k):
        clusterPoints.append([])
    
    return clusterPoints

  def calcCentroids():
    # print(pd.DataFrame(centroids))
    centroids = []
    for cluster in clusterPoints:
      data = pd.DataFrame(np.array(df)[[cluster]][0]).iloc[:, 1:]
      centroids.append(data.mean(axis=0))
    return centroids
  
  
  def getClasses():
    res = []
    for cluster in clusterPoints:
      data = pd.DataFrame(np.array(df)[[cluster]][0]).iloc[:, 0]
      print(cluster)
      res.append(data.mode()[0])
    return res
    
  clusterPoints = initiateClusterPoints()


  for _ in range(iterCount):
    clusterPoints = initiateClusterPoints()
    for row in df.iterrows():
      dist = []
      for centroid in centroids:
        dist.append(
          np.linalg.norm(
              np.array(centroid)-np.array(row[1][1:])
          )
        )
      sortInd = np.argsort(dist)
      clusterPoints[sortInd[0]] = np.append(
        clusterPoints[sortInd[0]], row[0]
      ).astype(int)

    centroids = calcCentroids()

    # print(pd.DataFrame(np.array(df)[[clusterPoints[0]]][0]))
    # print(clusterPoints)

  # print(centroids)
  classes = getClasses()
  print(classes)
  
      

if __name__ == '__main__':
  data = 0.0,0.0,-0.23636363636363636,0.04242424242424243,-0.48484848484848486,0.024242424242424242,-0.6727272727272727,0.03636363636363636,-0.8181818181818182,0.07878787878787878,-0.5636363636363636,-0.3212121212121212,-0.7696969696969697,-0.4303030303030303,-0.896969696969697,-0.4909090909090909,-1.0,-0.5515151515151515,-0.47878787878787876,-0.4303030303030303,-0.6787878787878788,-0.593939393939394,-0.806060606060606,-0.696969696969697,-0.9090909090909091,-0.7696969696969697,-0.3575757575757576,-0.49696969696969695,-0.503030303030303,-0.703030303030303,-0.6060606060606061,-0.8181818181818182,-0.696969696969697,-0.9090909090909091,-0.2,-0.5151515151515151,-0.23636363636363636,-0.7212121212121212,-0.2787878787878788,-0.8606060606060606,-0.3212121212121212,-0.9757575757575757
  kmeansModel("./dataset.csv")