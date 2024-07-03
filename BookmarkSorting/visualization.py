import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def create_cluster_visualization(reduced_features, labels, titles):
    df = pd.DataFrame(reduced_features[:, :2], columns=['Component 1', 'Component 2'])
    df['Cluster'] = labels
    df['Title'] = titles

    fig = px.scatter(df, x='Component 1', y='Component 2', color='Cluster', hover_data=['Title'],
                     title='HDBSCAN Clustering Results')
    return fig

def plot_dendrogram(linkage_matrix):
    fig = go.Figure(data=go.Dendrogram(root=dict(visible=False), orientation='left'))
    fig.update_layout(title='Hierarchical Clustering Dendrogram')
    return fig

def plot_treemap(linkage_matrix, labels, titles):
    n_clusters = min(10, len(set(labels)) - 1)  # Adjust the number of clusters, excluding noise
    
    if n_clusters > 1:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Create a mapping from original labels to hierarchical cluster labels
        label_mapping = {original: hierarchical for original, hierarchical in zip(range(len(cluster_labels)), cluster_labels)}
        
        df = pd.DataFrame({
            'Cluster': [label_mapping.get(label, 'Noise') if label != -1 else 'Noise' for label in labels],
            'SubCluster': ['Cluster ' + str(label) if label != -1 else 'Noise' for label in labels],
            'Title': titles
        })
    else:
        df = pd.DataFrame({
            'Cluster': ['Single Cluster'] * len(labels),
            'SubCluster': ['Cluster ' + str(label) if label != -1 else 'Noise' for label in labels],
            'Title': titles
        })
    
    df['Count'] = 1
    
    fig = px.treemap(df, path=['Cluster', 'SubCluster', 'Title'], values='Count',
                     title='Hierarchical Cluster Treemap')
    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    
    return fig