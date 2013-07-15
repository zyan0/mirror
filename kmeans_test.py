from kmeans import kMeans
from pyflann import *
from numpy import *
from numpy.random import *

dataset = [[1.0, 1.0], [1.1, 1.1], [0.1, 0.1], [0.0, 0.0]]

k = kMeans(dataset, 2, 10)
k.train()
print k.get_centers()
print k.predict([[0.0, 0.0], [1.0, 1.0], [0.9, 0.9], [-0.1, -0.1]])
