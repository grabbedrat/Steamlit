import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from bs4 import BeautifulSoup
import re

# Set page config
st.set_page_config(layout="wide", page_title="Bookmark Clustering")

# File upload
uploaded_file = st.file_uploader("Choose a bookmark HTML file", type="html")

if uploaded_file is not None:
    # Load and cache the file content
    @st.cache_data
    def load_file_content(file):
        return file.read().decode('utf-8')

    content = load_file_content(uploaded_file)
    soup = BeautifulSoup(content, 'html.parser')
    
    def extract_bookmarks(_soup):
        bookmarks = []
        for a in _soup.find_all('a'):
            bookmark = {
                'title': a.string,
                'url': a.get('href'),
                'add_date': a.get('add_date'),
                'tags': a.get('tags')
            }
            bookmarks.append(bookmark)
        return pd.DataFrame(bookmarks).rename(columns=lambda x: x.lower())

    # Create DataFrame
    bookmarks_df = extract_bookmarks(soup)
    
    # Data Preprocessing
    st.header("Data Preprocessing")
    with st.expander("Preprocessing Options", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            remove_duplicates = st.checkbox("Remove Duplicates", value=True)
            duplicate_criterion = st.radio("Duplicate Criterion", ["url", "title"])
        with col2:
            trim_titles = st.checkbox("Trim Long Titles", value=True)
            max_title_length = st.number_input("Max Title Length", min_value=10, max_value=200, value=100)

    # Perform preprocessing
    original_count = len(bookmarks_df)

    if remove_duplicates:
        bookmarks_df = bookmarks_df.drop_duplicates(subset=[duplicate_criterion], keep='first')

    if trim_titles:
        bookmarks_df['title'] = bookmarks_df['title'].apply(lambda x: x[:max_title_length] if x else x)

    preprocessed_count = len(bookmarks_df)
    st.write(f"Bookmarks reduced from {original_count} to {preprocessed_count}")
    
    # Display bookmarks and tags
    st.header("Bookmarks and Tags")
    with st.expander("View Bookmarks and Tags", expanded=False):
        # Create a dataframe with only title, url, and tags
        display_df = bookmarks_df[['title', 'url', 'tags']]
        
        # Add a search box
        search_term = st.text_input("Search bookmarks", "")
        
        if search_term:
            # Filter the dataframe based on the search term
            filtered_df = display_df[display_df['title'].str.contains(search_term, case=False) | 
                                     display_df['url'].str.contains(search_term, case=False) | 
                                     display_df['tags'].str.contains(search_term, case=False)]
        else:
            filtered_df = display_df
        
        # Display the filtered dataframe
        st.dataframe(filtered_df)
    
    # Embedding generation
    @st.cache_data
    def generate_embeddings(titles, urls):
        # Extract text from bookmarks (title and URL)
        texts = [f"{title} {url}" for title, url in zip(titles, urls)]
        
        # Clean texts
        texts = [re.sub(r'http\S+', '', text) for text in texts]
        texts = [re.sub(r'\s+', ' ', text).strip() for text in texts]
        
        # Generate embeddings using TF-IDF
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(texts).toarray()
        return embeddings

    embeddings = generate_embeddings(bookmarks_df['title'], bookmarks_df['url'])

    # Dimensionality reduction and clustering parameters
    with st.expander("Clustering Parameters", expanded=False):
        min_cluster_size = st.slider('Min Cluster Size', 2, 20, 5)
        min_samples = st.slider('Min Samples', 1, 10, 1)
        cluster_selection_epsilon = st.slider('Cluster Selection Epsilon', 0.0, 1.0, 0.0)
        metric = st.selectbox('Distance Metric', ['euclidean', 'cosine', 'manhattan'])

    with st.expander("Dimensionality Reduction", expanded=False):
        n_components = st.slider('Number of Components', 2, 10, 2)
        normalize = st.checkbox('Normalize Data', value=True)

    # Preprocessing and dimensionality reduction
    @st.cache_data
    def preprocess_and_reduce(embeddings, n_components, normalize):
        if normalize:
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)
        
        reducer = PCA(n_components=n_components)
        reduced_features = reducer.fit_transform(embeddings)
        return reduced_features

    reduced_features = preprocess_and_reduce(embeddings, n_components, normalize)

    # Clustering
    @st.cache_data
    def perform_clustering(reduced_features, min_cluster_size, min_samples, cluster_selection_epsilon, metric):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric
        )
        clusterer.fit(reduced_features)
        return clusterer

    clusterer = perform_clustering(reduced_features, min_cluster_size, min_samples, cluster_selection_epsilon, metric)

    # Visualization
    st.header('Clustering Visualization')
    df = pd.DataFrame(reduced_features[:, :2], columns=['Component 1', 'Component 2'])
    df['Cluster'] = clusterer.labels_
    df['Title'] = bookmarks_df['title']

    fig = px.scatter(df, x='Component 1', y='Component 2', color='Cluster', hover_data=['Title'],
                     title='HDBSCAN Clustering Results')
    st.plotly_chart(fig, use_container_width=True)

    # Display cluster contents
    st.header('Cluster Contents')
    cluster_df = pd.DataFrame({
        'Title': bookmarks_df['title'],
        'URL': bookmarks_df['url'],
        'Cluster': clusterer.labels_
    })

    for cluster in sorted(set(clusterer.labels_)):
        if cluster == -1:
            st.subheader('Noise Points')
        else:
            st.subheader(f'Cluster {cluster}')
        
        cluster_items = cluster_df[cluster_df['Cluster'] == cluster][['Title', 'URL']]
        st.dataframe(cluster_items)

    # Generate prompts
    st.header('Generated Prompts')

    def generate_prompts(cluster_id, indent=""):
        prompts = []
        folder_name = f"Folder {cluster_id}"
        bookmarks_in_cluster = bookmarks_df[clusterer.labels_ == cluster_id]
        
        if not bookmarks_in_cluster.empty:
            bookmark_titles = bookmarks_in_cluster['title'].tolist()
            bookmark_titles_str = "\n".join([f"{indent}  - {title}" for title in bookmark_titles])
            prompt = f"{indent}Folder: {folder_name}\n{bookmark_titles_str}"
            prompts.append(prompt)
        
        return prompts

    all_prompts = []
    for cluster in sorted(set(clusterer.labels_)):
        if cluster != -1:  # Exclude noise points
            cluster_prompts = generate_prompts(cluster)
            all_prompts.extend(cluster_prompts)
            for prompt in cluster_prompts:
                st.text(prompt)
                st.text("")  # Add an empty line for readability

    # Download prompts
    if all_prompts:
        prompt_text = "\n\n".join(all_prompts)
        st.download_button(
            label="Download Prompts",
            data=prompt_text,
            file_name="prompts.txt",
            mime="text/plain"
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