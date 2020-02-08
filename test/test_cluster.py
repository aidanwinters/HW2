from hw2 import cluster
from hw2 import io
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test_partition_clustering():

    #to test, I created a data set of two features with 4 clusterings
    x = np.append(np.random.binomial(10, 0.5, 50) + 50, np.random.binomial(10, 0.5, 50))
    y1 = np.append(np.random.binomial(10, 0.5, 25), np.random.binomial(10, 0.5, 25) + 50)
    y = np.append(y1, y1)
    print(x,y)

    dat = pd.DataFrame({'x':x, 'y':y})

    #test if clustering is done correctly and produces expected labels
    labels = cluster.kmeans(dat, 4)
    assert len(labels) == 4 #checks if size of returned dictionary is correct

    #To test this, I will iterate over some randomly produced vectors similar to my features
    for k in range(10):
        x = [random.uniform(0, 1) for x in range(20)]
        y = [random.uniform(0, 1) for x in range(20)]

        dist = cluster.euclidean_distance(x, y)

        assert cluster.euclidean_distance(x, x) == 0 # dist(a,a)==0
        assert dist == cluster.euclidean_distance(y, x)# dist(a,b)==dist(b,a)
        assert dist >= 0 # sign(dist(a,b))==+

    return

def test_hierarchical_clustering():

    #to test, I created a data set of two features with 4 clusterings
    x = np.append(np.random.binomial(10, 0.5, 50) + 50, np.random.binomial(10, 0.5, 50))
    y1 = np.append(np.random.binomial(10, 0.5, 25), np.random.binomial(10, 0.5, 25) + 50)
    y = np.append(y1, y1)
    print(x,y)

    dat = pd.DataFrame({'x':x, 'y':y})

    dist = cluster.get_distances(dat)

    #test if clustering is done correctly and produces expected labels
    labels = cluster.merge(dist, 4, 'single')
    assert len(labels) == 4 #checks if size of returned dictionary is correct

    #To test this, I will iterate over some randomly produced vectors similar to my features
    for k in range(10):
        x = [random.uniform(0, 1) for x in range(20)]
        y = [random.uniform(0, 1) for x in range(20)]

        dist = cluster.euclidean_distance(x, y)

        assert cluster.euclidean_distance(x, x) == 0 # dist(a,a)==0
        assert dist == cluster.euclidean_distance(y, x)# dist(a,b)==dist(b,a)
        assert dist >= 0 # sign(dist(a,b))==+

    return
