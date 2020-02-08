import sys
from .io import read_active_sites, write_clustering, write_mult_clusterings
from .cluster import cluster_by_partitioning, cluster_hierarchically, get_residue_features, silhouette, get_distances

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 4:
    print("Usage: python -m hw2 [-P| -H] <pdb directory> <output file>")
    sys.exit(0)

active_sites = read_active_sites(sys.argv[2])

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    clustering = cluster_by_partitioning(active_sites)
    write_clustering(sys.argv[3], clustering)

if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    clusterings = cluster_hierarchically(active_sites)
    write_clustering(sys.argv[3], clusterings)

if sys.argv[1][0:2] == '-T':
    print("Comparing Clustering on Active Sites")

    feats = get_residue_features(active_sites)

    lbl_kmeans = cluster_by_partitioning(active_sites)
    lbl_hier = cluster_hierarchically(active_sites)

    k_score = silhouette(feats, lbl_kmeans)
    h_score = silhouette(feats, lbl_hier)

    print('Score with 10 clusters:')
    print('k means: ', k_score)
    print('Hierarchical: ', h_score)
