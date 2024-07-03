import streamlit as st
import plotly.express as px
import pandas as pd

def create_cluster_visualization(reduced_features, labels, titles):
    df = pd.DataFrame(reduced_features[:, :2], columns=['Component 1', 'Component 2'])
    df['Cluster'] = labels
    df['Title'] = titles

    fig = px.scatter(df, x='Component 1', y='Component 2', color='Cluster', hover_data=['Title'],
                     title='HDBSCAN Clustering Results')
    return fig

def display_cluster_contents(labels, bookmarks_df):
    cluster_df = pd.DataFrame({
        'Title': bookmarks_df['title'],
        'URL': bookmarks_df['url'],
        'Cluster': labels
    })

    for cluster in sorted(set(labels)):
        if cluster == -1:
            st.subheader('Noise Points')
        else:
            st.subheader(f'Cluster {cluster}')
        
        cluster_items = cluster_df[cluster_df['Cluster'] == cluster][['Title', 'URL']]
        st.dataframe(cluster_items)