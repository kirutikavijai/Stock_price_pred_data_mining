import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

def gap_statistic(X, k_max=10, n_refs=10, random_state=42):
    """
    Computes Gap Statistic for KMeans clustering.

    X           : normalized feature matrix
    k_max       : maximum K to test
    n_refs      : number of reference datasets
    random_state: reproducible results
    """

    np.random.seed(random_state)

    gaps = np.zeros(k_max)
    s_k = np.zeros(k_max)
    Wks = np.zeros(k_max)
    Wkbs = np.zeros((k_max, n_refs))

    # Bounds for generating reference datasets
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    def compute_wk(X, labels):
        """Within-cluster dispersion"""
        k = len(np.unique(labels))
        Wk = 0
        for cluster in np.unique(labels):
            subset = X[labels == cluster]
            if len(subset) > 1:
                distances = pairwise_distances(subset)
                Wk += np.sum(distances) / (2 * len(subset))
        return Wk

    for k in range(1, k_max + 1):
        # Fit KMeans on actual data
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(X)
        Wk = compute_wk(X, labels)
        Wks[k - 1] = np.log(Wk)

        # Reference datasets
        for i in range(n_refs):
            X_ref = np.random.uniform(mins, maxs, size=X.shape)
            km_ref = KMeans(n_clusters=k, random_state=random_state)
            ref_labels = km_ref.fit_predict(X_ref)
            Wk_ref = compute_wk(X_ref, ref_labels)
            Wkbs[k - 1, i] = np.log(Wk_ref)

        # Compute Gap(k)
        gaps[k - 1] = np.mean(Wkbs[k - 1]) - np.log(Wk)
        s_k[k - 1] = np.std(Wkbs[k - 1]) * np.sqrt(1 + 1/n_refs)

    return gaps, s_k, Wks

import numpy as np
from sklearn.metrics import pairwise_distances

def dunn_index(X, labels):
    """
    Compute Dunn Index for clustering.

    X : array-like (n_samples, n_features)
    labels : array of cluster labels
    """

    # Unique clusters (ignore noise = -1 if any)
    clusters = np.unique(labels)
    clusters = clusters[clusters != -1]

    # Compute distance matrix only once
    distances = pairwise_distances(X)

    # Intra-cluster distances (cluster diameters)
    intra_dists = []
    for c in clusters:
        points = np.where(labels == c)[0]
        if len(points) > 1:
            # max distance between any two points in the same cluster
            intra_dist = np.max(distances[np.ix_(points, points)])
            intra_dists.append(intra_dist)

    max_intra = np.max(intra_dists)

    # Inter-cluster distances
    inter_dists = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            c1 = np.where(labels == clusters[i])[0]
            c2 = np.where(labels == clusters[j])[0]

            # minimum distance between any two points in clusters i and j
            inter = np.min(distances[np.ix_(c1, c2)])
            inter_dists.append(inter)

    min_inter = np.min(inter_dists)

    # Dunn Index
    return min_inter / max_intra
