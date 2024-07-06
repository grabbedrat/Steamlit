import streamlit as st
import pandas as pd
import numpy as np
import time
from data_preprocessing import load_and_preprocess_data
from embedding import generate_embeddings
from clustering import preprocess_and_reduce, perform_clustering, perform_hierarchical_clustering, plot_silhouette
from visualization import create_cluster_visualization, plot_dendrogram, plot_treemap, create_minimum_spanning_tree
from utils import generate_prompts, perform_lsa, generate_html_structure
from folder_naming import name_folders, update_folder_names

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

    # LSA parameters
    with st.expander("LSA Parameters", expanded=False):
        n_components = st.slider('Number of LSA Components', min_value=2, max_value=100, value=50, key="n_components",
                                help="Number of concepts to extract using LSA. Higher values preserve more information but may include noise.")

    # Embedding generation
    lsa_matrix = perform_lsa(bookmarks_df, n_components)

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

    reduced_features = lsa_matrix

    # Clustering
    clusterer, silhouette_avg = perform_clustering(reduced_features, min_cluster_size, min_samples, cluster_selection_epsilon, metric)

    if silhouette_avg != -1:
        st.write(f"Silhouette Score: {silhouette_avg:.4f}")

        # Silhouette Plot
        st.subheader("Silhouette Plot")
        silhouette_fig = plot_silhouette(reduced_features, clusterer.labels_)
        if silhouette_fig:
            st.pyplot(silhouette_fig)
    else:
        st.warning("Could not calculate silhouette score. There might not be enough clusters or non-noise points.")

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

    # Minimum Spanning Tree Visualization
    st.header('Minimum Spanning Tree')
    mst_fig = create_minimum_spanning_tree(reduced_features, clusterer.labels_, bookmarks_df['title'])
    st.plotly_chart(mst_fig, use_container_width=True)

    # Generate LLM prompts and hierarchy
    llm_prompts, hierarchy = generate_prompts(clusterer.labels_, bookmarks_df, linkage_matrix)

    # Generate new folder names
    new_folder_names = name_folders(llm_prompts)

    # Update the hierarchy with new folder names
    updated_hierarchy = update_folder_names(hierarchy, new_folder_names)

    # Generate HTML structure with updated folder names
    current_timestamp = int(time.time())
    updated_hierarchical_html = generate_html_structure(updated_hierarchy, current_timestamp)

    # Display the original and new folder names
    st.subheader("Folder Naming Results")
    for old_folder_name, folder_info in new_folder_names.items():
        new_folder_name = folder_info['name']
        st.write(f"Folder: {old_folder_name} → {new_folder_name}")
        for old_subfolder_name, new_subfolder_name in folder_info['subfolders'].items():
            st.write(f"  Subfolder: {old_subfolder_name} → {new_subfolder_name}")

    # Add a section to show the content of each renamed folder
    st.subheader("Folder Contents")
    for old_folder_name, folder_info in new_folder_names.items():
        new_folder_name = folder_info['name']
        with st.expander(f"{new_folder_name} (was: {old_folder_name})"):
            if 'subfolders' in llm_prompts[old_folder_name]:
                for old_subfolder_name, subfolder_info in llm_prompts[old_folder_name]['subfolders'].items():
                    new_subfolder_name = folder_info['subfolders'].get(old_subfolder_name, old_subfolder_name)
                    st.write(f"Subfolder: {new_subfolder_name} (was: {old_subfolder_name})")
                    st.write("\n".join(f"  - {item}" for item in subfolder_info['content']))
                    st.write("")
            else:
                st.write("\n".join(f"- {item}" for item in llm_prompts[old_folder_name]['content']))

    # Generate HTML structure with updated folder names
    current_timestamp = int(time.time())
    updated_hierarchical_html = generate_html_structure(updated_hierarchy, current_timestamp)

    # Display a preview of the updated HTML structure
    st.text_area("Preview of Updated Hierarchical Structure", updated_hierarchical_html[:1000] + "...", height=300)

    # Download updated HTML file
    st.download_button(
        label="Download Updated Hierarchical Bookmarks HTML",
        data=updated_hierarchical_html,
        file_name="updated_clustered_bookmarks.html",
        mime="text/html",
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