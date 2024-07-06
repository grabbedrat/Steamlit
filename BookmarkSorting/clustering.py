import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import streamlit as st
from scipy.cluster.hierarchy import linkage

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
    
    # Calculate silhouette score only for non-noise points
    non_noise_mask = clusterer.labels_ != -1
    if np.sum(non_noise_mask) > 1:  # Ensure there are at least 2 non-noise points
        silhouette_avg = silhouette_score(reduced_features[non_noise_mask], 
                                          clusterer.labels_[non_noise_mask])
    else:
        silhouette_avg = -1  # Invalid silhouette score

    return clusterer, silhouette_avg

def perform_hierarchical_clustering(clusterer, reduced_features):
    unique_labels = np.unique(clusterer.labels_)
    cluster_centers = np.array([reduced_features[clusterer.labels_ == label].mean(axis=0) for label in unique_labels if label != -1])
    linkage_matrix = linkage(cluster_centers, method='ward')
    return linkage_matrix

def calculate_silhouette_score(reduced_features, labels):
    if len(set(labels)) <= 1:
        return -1  # Silhouette Score is not defined for a single cluster
    return silhouette_score(reduced_features, labels)

def plot_silhouette(reduced_features, labels):
    non_noise_mask = labels != -1
    non_noise_features = reduced_features[non_noise_mask]
    non_noise_labels = labels[non_noise_mask]
    n_clusters = len(set(non_noise_labels))
    
    if n_clusters <= 1:
        st.warning("Not enough clusters to calculate silhouette score.")
        return None

    silhouette_avg = silhouette_score(non_noise_features, non_noise_labels)
    sample_silhouette_values = silhouette_samples(non_noise_features, non_noise_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[non_noise_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("Silhouette plot for the various clusters")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.tight_layout()
    return fig