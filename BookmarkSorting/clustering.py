import streamlit as st
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

@st.cache_data
def preprocess_and_reduce(embeddings, n_components, normalize):
    if normalize:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    reducer = PCA(n_components=n_components)
    reduced_features = reducer.fit_transform(embeddings)
    return reduced_features

@st.cache_data
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
        clusterer.fit(distance_matrix)
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric
        )
        clusterer.fit(reduced_features)
    
    return clusterer