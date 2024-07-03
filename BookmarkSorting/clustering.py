import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

def preprocess_and_reduce(embeddings, n_components, normalize, method='PCA'):
    if normalize:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    reduced_features = reducer.fit_transform(embeddings)
    return reduced_features.astype(np.float64)  # Ensure float64 type

def perform_clustering(reduced_features, min_cluster_size, min_samples, cluster_selection_epsilon, metric):
    if metric == 'cosine':
        # Precompute cosine distance matrix
        distance_matrix = cosine_distances(reduced_features)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='precomputed'
        )
        clusterer.fit(distance_matrix.astype(np.float64))  # Ensure float64 type
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric
        )
        clusterer.fit(reduced_features.astype(np.float64))  # Ensure float64 type
    
    return clusterer

def perform_hierarchical_clustering(clusterer, reduced_features):
    from scipy.cluster.hierarchy import linkage
    
    unique_labels = np.unique(clusterer.labels_)
    cluster_centers = np.array([reduced_features[clusterer.labels_ == label].mean(axis=0) for label in unique_labels if label != -1])
    linkage_matrix = linkage(cluster_centers, method='ward')
    return linkage_matrix