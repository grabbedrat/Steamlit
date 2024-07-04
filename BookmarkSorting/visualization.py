import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx
from scipy.spatial.distance import pdist, squareform

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

def create_minimum_spanning_tree(reduced_features, labels, titles):
    # Calculate pairwise distances
    distances = pdist(reduced_features)
    dist_matrix = squareform(distances)

    # Create a graph
    G = nx.from_numpy_array(dist_matrix)

    # Calculate the minimum spanning tree
    mst = nx.minimum_spanning_tree(G)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in mst.edges():
        x0, y0 = reduced_features[edge[0]][:2]
        x1, y1 = reduced_features[edge[1]][:2]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = reduced_features[:, 0]
    node_y = reduced_features[:, 1]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Cluster',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Color nodes by cluster and add hover text
    node_trace.marker.color = labels
    node_trace.text = titles

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Minimum Spanning Tree',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig