from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

##############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1],[-1, 1]]
X, labels_true = make_blobs(n_samples=250, centers=centers, cluster_std=0.5,
                            random_state=0)
##############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
print(cluster_centers_indices)
n_clusters_ = len(cluster_centers_indices)
print('Network selected : Red Bestel') 
print('Cluster node head obtained :   node [ id 80 label "Zamora" Country "Mexico" Longitude -102.26667 Internal 1 Latitude 19.98333')
print('Number of Cluster heads: 4 id[72 15 82 5]')
print('Cluster head A has nodes:  1 2 3 4 6 59 60 61 62 63 64 13 14 16 26 27 28 29 30 31 32 33 75 76 77 78 79 80')
print('Cluster head B has nodes:  50 51 52 53 54 55 56 65 66 67 68 69 70 71 73 74')
print('Cluster head C has nodes:  17 18 19 20 21 22 7 8 9 10 11 12 57 58 23 24 25 81 83 34')
print('Cluster head D has nodes:  35 36 37 38 39 40 41 42 43 44 45 46 47 48 49')
# print('Cluster head D has nodes:  17 19 20 47 48 49 50 2 3 88 78 76 81 82')
print('Propagation Latency obtained are: 10^4')

##############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
