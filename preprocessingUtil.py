import copy
import itertools

def normalizeToBase(landmarks):
  res = copy.deepcopy(landmarks)
  basex, basey = 0,0

  for index, landpoint in enumerate(res):
    if index == 0:
      basex, basey = landpoint[0], landpoint[1]
    
    res[index][0] = res[index][0] - basex
    res[index][1] = res[index][1] - basey

  res = list(itertools.chain.from_iterable(res))

  maxVal = max(list(map(abs, res)))

  def norm(n):
    return n/maxVal

  res = list(map(norm, res))

  return res

def preprocessZ(zpoint, zqueue):
  zpoint = abs(zpoint)
  t = (zpoint-0.02)/(0.4-0.02)
  t *= 100
  zqueue.append(abs(t))
  sum = 0
  for i in zqueue:
    if i:
      sum+=i
  t = sum/len(zqueue)
  return t, zqueue

def normalizeXY(loc, xqueue, yqueue):
  xqueue.append(loc[0]/2)
  sumX = 0
  for i in xqueue:
    if i:
      sumX+=i
  x = sumX/len(xqueue)

  yqueue.append(loc[1])
  sumY = 0
  for i in yqueue:
    if i:
      sumY+=i
  y = sumY/len(yqueue)

  # print(int(x)*2, int(y))

  return [int(x)*2, int(y)], xqueue, yqueue