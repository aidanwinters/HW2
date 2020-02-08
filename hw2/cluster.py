from .utils import Atom, Residue, ActiveSite
import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sklearn

def get_residue_features(active_sites):

    template = {}

    df = pd.DataFrame(columns=['ASP', 'SER', 'ARG', 'LYS', 'GLU', 'THR', 'TYR', 'ASN', 'PHE', 'TRP', 'HIS', 'GLY', 'LEU', 'CYS', 'PRO', 'ALA', 'ILE'])


    for act in active_sites:
        act_name = str(act.name)
        df.loc[act_name] = 0
        types = [res.type for res in act.residues]
        for t in types:
            df.loc[act_name, t] += 1
        df.loc[act_name] = df.loc[act_name]/len(types)

    return(df)


def get_umap(df):
    return umap.UMAP().fit(df)

def cluster_by_partitioning(active_sites):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    # Fill in your code here!

    feats = get_residue_features(active_sites)

    #CLUSTER ON UMAP
    # u = get_umap(feats)

    #
    # k = 5
    # labels, centr = kmeans(pd.DataFrame(u.embedding_), k)
    #
    # for ind in range(len(labels)):
    #     plt.scatter(u.embedding_[:, 0], u.embedding_[:, 1], c=labels[ind], s=6**2)
    #     plt.scatter(centr[ind].loc[:,0],centr[ind].loc[:,1], c=list(set(labels[ind])), s=12**2, marker = "x")
    #     plt.show()
    #
    #CLUSTER ON FEATS
    labels = kmeans(feats, 10)

    return labels

    # u = get_umap(feats)
    # for ind in range(len(labels)):
    #     plt.scatter(u.embedding_[:, 0], u.embedding_[:, 1], c=labels[ind], s=6**2)
    #     centr_embed = u.transform(centr[ind])
    #     plt.scatter(centr_embed[:, 0], centr_embed[:, 1], c="red", s=12**2, marker = "x")
    #     plt.show()
    #
    #
    # return labels

def all_equal(iterator):
  try:
     iterator = iter(iterator)
     first = next(iterator)
     return all(np.array_equal(first, rest) for rest in iterator)
  except StopIteration:
     return True

def kmeans(feats, k, iter = 10, prev_check = 1):

    label_arr = []
    centr_arr = []

    #maybe add in recording the centroid so we can see animation of it?

    centr = init_centroids(feats, k)

    #store labels in a data frame or matrix

    for i in range(iter):
        #calculate distance of each point to each centroid
        dist = get_distances(feats, centr)

        labels = np.argmin(dist, axis=1) #get labels, which are really just the minimum column
        label_arr.append(labels) #save labels
        centr_arr.append(centr) #save labels

        #check if we are past the number of previous labels to check for changes
        #if there have already been prev_check iterations, we can do a check to see if the labels have stabilized
        if(i >= prev_check):
            #if the length of the set of arrays is 1, then all elements are equal, meaning all labels are the same
            if(all_equal(label_arr[i-prev_check:i])):
                break #if this is true, return

        #update centroids
        centr = get_centroids(feats, labels)

    #return only final labels

    fnl = label_arr[len(label_arr) - 1]
    final_lbl = [[] for x in set(fnl)]
    for l in range(len(fnl)):
        final_lbl[fnl[l]].append(l)

    print(final_lbl)

    return final_lbl

#find centroids by randomly sampling from possible values for each features
#this ensures that our centroids are relatively close to data points
def init_centroids(feats, k):

    c = []

    for x in range(k):
        c.append([random.choice(feats.iloc[:,col]) for col in range(len(feats.columns))])

    return pd.DataFrame(c)


def get_centroids(feats, labels):

    #ensure it is a numpy arr
    labels = np.array(labels)

    updated = []

    #iterate through labels
    for lab in set(labels):
        inds = np.where( labels == lab)[0]
        rows = feats.iloc[inds]

        updated.append(rows.sum(0)/len(rows))

    #for now just initialize a random one within the range of the features
    return pd.DataFrame(updated)

def get_distances(matr1, matr2=None):

    if type(matr2) == type(None): # in this case we can run a slightly faster uper triangle only comparison

        dist = np.zeros((len(matr1), len(matr1)))
        for i in range(len(matr1)):
            temp = []
            for j in range(i + 1, len(matr1)): #only the vectors not already compared

                d = euclidean_distance(matr1.iloc[i], matr1.iloc[j])
                dist[i, j], dist[j, i] = d , d

    else:
        dist = np.zeros((len(matr1), len(matr2)))
        for i in range(len(matr1)):
            temp = []
            for j in range(len(matr2)): #only the vectors not already compared
                dist[i, j] = euclidean_distance(matr1.iloc[i], matr2.iloc[j])

    return dist

def euclidean_distance(a, b, power=2):
    a = np.array(a)
    b = np.array(b)
    return sum((a - b)**power)**(1/power)

#get minimum from upper triangle of pandas
def get_min_upper(df):

    #assumes the matrix is square
    min  = float("inf")
    ind = (-1,-1)

    for i in range(0, len(df)):
        for j in range(i + 1, len(df)):
            if(df[i][j] < min):
                min = df[i][j]
                ind = (i,j)

    return ind[0], ind[1]


def cluster_hierarchically(active_sites):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    feats = get_residue_features(active_sites)

    dist = get_distances(feats)

    lbl_dict = merge(dist, 10, 'single')

    final_lbl = []
    for key in lbl_dict:
        final_lbl.append(lbl_dict[key])

    print(final_lbl)

    return final_lbl


#here is where you can add new link functions
#currently this is just single and complete
def link(a, b, link_metric):
    if (link_metric == "single"):
        return min(a,b)
    elif (link_metric == "complete"):
        return max(a, b)
    else:
        raise TypeError('Link not recognized.')
        return

#this function calls a second merge function until the points
#have been placed into the desired number of clusters
def merge(dist, numClusters, metric):

    clusterNames = np.arange(0, len(dist))
    clust_dict = {}

    #create dictionary to record clusters
    for k in range(len(dist)):
        clust_dict[k] = [k]

    #iterate until the size of the matrix (which shrinks) is less than the numClusters
    #this means we have finished clustering
    while len(dist) > numClusters:
        dist, clust_dict = merge_closest(dist, metric, clust_dict)

    return clust_dict

#identify and then merge the two closest points
#return a new distance matrix with the merged points
def merge_closest(distance, metric, clust_dict):

    i, j = get_min_upper(distance)

    #combine two into cluster in dictionary
    #drop key j from dictinonary
    clust_dict[i] = clust_dict[i] + clust_dict[j]
    clust_dict.pop(j, None)

    #iterate through distance matrix and store the new distance to the new cluster
    #in the position of i
    for k in range(len(distance)):
        if (k != i and k != j):
            distance[k][i] = link(distance[k][i], distance[k][j], metric)

    #remove the column and row for position j
    distance = np.delete(distance, j, axis=0)
    distance = np.delete(distance, j, axis=1)


    #reformat dict so that key range matches the number of columns in distance
    #this maintins the keys as direct indices for our distance matrix
    reformat_d = {}
    iter = 0
    r = range(len(clust_dict))
    for key in sorted (clust_dict.keys()):
        reformat_d[iter] = clust_dict[key]
        iter += 1

    return distance, reformat_d


#compute the silhouette score for a feature set and list of clusters:
def silhouette(feats, labels):

    distance = get_distances(feats)

    n = len(labels)

    flat_lbl = np.zeros(len(distance))

    for l in range(len(labels)):
        for x in labels[l]:
            flat_lbl[x] = l

    return sklearn.metrics.silhouette_score(distance, flat_lbl)

    #
    # for clust in labels:
    #     #sum only the points within cluster
    #     sum = np.sum(distance.iloc[clust,clust])
    #
    #
    #     for c in clust:
    #         sum feats[]
    #
    # sums = np.sum([[df[index[u]][index[v]] for u in x] for v in x])
    # div_square = [x / (len())]
    #
    # intraCluster = [(np.sum([[df[index[u]][index[v]] for u in x] for v in x
    #                      ])) / (len(x) ** 2) for x in clusterList] #intracluster similarity computation #intracluster similarity computation
    #
    # score = 0 #initialization
    # for i in range(0, len(clusterList)): #for all submitted clusters
    #     for j in range(i, len(clusterList)): #nested ticker
    #         score += np.sqrt(intraCluster[i] * intraCluster[j]) * ( #* intercluster dissimilarity
    #             1.0 - np.sum([[df[index[u]][index[v]] for u in clusterList[i]] for v in clusterList[j]]
    #                             ) / (len(clusterList[i]) + len(clusterList[j]))) #add the calculated score and normalize
    #
    # score = ((1 / (n * (n-1))) * score) / 1000 #compute the comparison value normalized to the amount of clusters

    return score
