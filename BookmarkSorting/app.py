import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessing import load_and_preprocess_data
from embedding import generate_embeddings
from clustering import preprocess_and_reduce, perform_clustering, perform_hierarchical_clustering
from visualization import create_cluster_visualization, plot_dendrogram, plot_treemap
from utils import generate_prompts

# Set page config
st.set_page_config(layout="wide", page_title="Bookmark Clustering")

# File upload
uploaded_file = st.file_uploader("Choose a bookmark HTML file", type="html", key="file_uploader")

if uploaded_file is not None:
    # Load and preprocess data
    if 'bookmarks_df' not in st.session_state:
        st.session_state.bookmarks_df, st.session_state.tagging_info = load_and_preprocess_data(uploaded_file)
    
    bookmarks_df = st.session_state.bookmarks_df
    tagging_info = st.session_state.tagging_info
    
    # Display bookmarks and tags
    st.header("Bookmarks and Tags")
    with st.expander("View Bookmarks and Tags", expanded=False):
        display_df = bookmarks_df[['title', 'url', 'tags']]
        search_term = st.text_input("Search bookmarks", "", key="search_bookmarks")
        
        if search_term:
            filtered_df = display_df[display_df['title'].str.contains(search_term, case=False) | 
                                     display_df['url'].str.contains(search_term, case=False) | 
                                     display_df['tags'].str.contains(search_term, case=False)]
        else:
            filtered_df = display_df
        
        st.dataframe(filtered_df, key="filtered_df")
    
    # Embedding generation
    embeddings = generate_embeddings(bookmarks_df['title'], bookmarks_df['url'], bookmarks_df['tags'])

    # Clustering parameters
    with st.expander("Clustering Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            min_cluster_size = st.slider('Min Cluster Size', min_value=2, max_value=20, value=3, key="min_cluster_size",
                                         help="Minimum number of samples in a cluster. Smaller values allow for more fine-grained clusters.")
            
            min_samples = st.slider('Min Samples', min_value=1, max_value=10, value=1, key="min_samples",
                                    help="Number of samples in a neighborhood for a point to be considered a core point. Higher values make the algorithm more conservative.")
            
        with col2:
            cluster_selection_epsilon = st.slider('Cluster Selection Epsilon', min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="cluster_selection_epsilon",
                                                  help="Distance threshold for cluster merging. Smaller values allow for more fine-grained clusters. Use 0.0 for automatic selection.")
            
            metric = st.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine'], index=2, key="distance_metric",
                                  help="Method to calculate distance between points. 'cosine' is often good for text-based data.")

    # Dimensionality reduction parameters
    with st.expander("Dimensionality Reduction", expanded=False):
        dimensionality_reduction_method = st.selectbox('Dimensionality Reduction Method', ['PCA', 'UMAP', 't-SNE'], index=1, key="dimensionality_reduction_method",
                                                       help="Method to reduce the dimensionality of the data. UMAP and t-SNE often produce better separations for visualization.")
        
        n_components = st.slider('Number of Components', min_value=2, max_value=50, value=10, key="n_components",
                                 help="Number of dimensions to reduce the data to. Higher values preserve more information.")
        
        normalize = st.checkbox('Normalize Data', value=True, key="normalize_data",
                                help="Standardize features by removing the mean and scaling to unit variance.")

    # Preprocessing and dimensionality reduction
    reduced_features = preprocess_and_reduce(embeddings, n_components, normalize, dimensionality_reduction_method)

    # Clustering
    clusterer = perform_clustering(reduced_features, min_cluster_size, min_samples, cluster_selection_epsilon, metric)

    # Visualization
    st.header('Cluster Visualization and Hierarchy')
    col1, col2 = st.columns(2)

    with col1:
        fig = create_cluster_visualization(reduced_features, clusterer.labels_, bookmarks_df['title'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        unique_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        if unique_clusters > 1:
            linkage_matrix = perform_hierarchical_clustering(clusterer, reduced_features)
            treemap_fig = plot_treemap(linkage_matrix, clusterer.labels_, bookmarks_df['title'])
            st.plotly_chart(treemap_fig, use_container_width=True)
        else:
            st.write("Not enough clusters to create a hierarchy. Try adjusting clustering parameters.")

    # Generate prompts
    st.header('Generated Prompts')
    all_prompts = generate_prompts(clusterer.labels_, bookmarks_df)
    
    for i, prompt in enumerate(all_prompts):
        st.text_area(f"Prompt {i+1}", prompt, height=100, key=f"prompt_{i}")
        st.text("")  # Add an empty line for readability

    # Download prompts
    if all_prompts:
        prompt_text = "\n\n".join(all_prompts)
        st.download_button(
            label="Download Prompts",
            data=prompt_text,
            file_name="prompts.txt",
            mime="text/plain",
            key="download_prompts"
        )

else:
    st.write("Please upload an HTML file containing your bookmarks.")

# Add instructions for exporting bookmarks from Firefox
st.sidebar.header("How to export bookmarks from Firefox")
st.sidebar.markdown("""
1. Open Firefox and click on the Library button (book icon) in the toolbar.
2. Select "Bookmarks" and then "Show All Bookmarks".
3. In the Library window, click on "Import and Backup" in the toolbar.
4. Select "Backup...".
5. Choose a location to save the file and click "Save".
6. Upload the saved HTML file using the file uploader above.
""")