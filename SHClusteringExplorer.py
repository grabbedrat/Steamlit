import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Advanced Semantic Hierarchical Clustering Explorer")

# Sidebar for input and parameters
with st.sidebar:
    st.header("Input and Parameters")
    sentences = st.text_area("Enter sentences (one per line):",
    "The cat sat on the mat.\n"
    "Dogs chase cats.\n"
    "I love pizza.\n"
    "Pepperoni is my favorite topping.\n"
    "Artificial intelligence is fascinating.\n"
    "Machine learning models can be complex.\n"
    "The sky is blue.\n"
    "Clouds are white and fluffy.")
    
    sentence_list = sentences.split('\n')
    
    model_name = st.selectbox("Select Sentence Transformer model:", 
                              ['all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2'])
    
    linkage_method = st.selectbox("Select linkage method:", 
                                  ['ward', 'complete', 'average', 'single'])

@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

model = load_model(model_name)

embeddings = model.encode(sentence_list)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cluster Analysis")
    n_clusters = st.slider("Number of clusters:", min_value=2, max_value=len(sentence_list)-1, value=3)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(embeddings)
    
    df = pd.DataFrame({'Sentence': sentence_list, 'Cluster': cluster_labels})
    st.dataframe(df)
    
    # Silhouette score
    sil_score = silhouette_score(embeddings, cluster_labels)
    st.metric("Silhouette Score", f"{sil_score:.3f}")

with col2:
    st.subheader("Word Cloud by Cluster")
    cluster = st.selectbox("Select cluster to visualize:", range(n_clusters))
    
    cluster_text = ' '.join(df[df['Cluster'] == cluster]['Sentence'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

st.subheader("Dendrogram")
linkage_matrix = linkage(embeddings, method=linkage_method)
fig = ff.create_dendrogram(embeddings, labels=sentence_list)
fig.update_layout(height=600, title=f"Sentence Clustering Dendrogram ({linkage_method} linkage)")

for i in range(n_clusters):
    fig.add_vline(x=len(sentence_list) - n_clusters + i + 0.5, line_dash="dash", line_color="red")

st.plotly_chart(fig, use_container_width=True)

st.subheader("Semantic Similarity Heatmap")
similarity_matrix = np.inner(embeddings, embeddings)
fig = px.imshow(similarity_matrix,
                labels=dict(x="Sentence", y="Sentence", color="Similarity"),
                x=sentence_list, y=sentence_list,
                color_continuous_scale="Viridis")
fig.update_layout(height=600, title="Sentence Similarity Heatmap")
st.plotly_chart(fig, use_container_width=True)

# Interactive 3D scatter plot of sentence embeddings
st.subheader("3D Visualization of Sentence Embeddings")
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

fig = px.scatter_3d(x=embeddings_3d[:, 0], y=embeddings_3d[:, 1], z=embeddings_3d[:, 2],
                    color=cluster_labels, text=sentence_list,
                    labels={'color': 'Cluster'})
fig.update_traces(textposition='top center')
fig.update_layout(height=700, title="3D PCA of Sentence Embeddings")
st.plotly_chart(fig, use_container_width=True)

# Add interesting facts and explanations
st.subheader("Insights and Explanations")
st.markdown("""
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters. 
  Ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster.
- **Word Cloud**: Visualizes the most frequent words in the selected cluster, giving insight into cluster themes.
- **Dendrogram**: Shows the hierarchical relationship between sentences. The height of each 'U' represents 
  the distance between the two connected points.
- **Similarity Heatmap**: Darker colors indicate higher similarity between sentences.
- **3D Visualization**: Uses PCA to reduce embeddings to 3 dimensions, allowing us to visualize cluster separation.
""")

facts = [
    "Hierarchical clustering can be performed using either a bottom-up (agglomerative) or top-down (divisive) approach.",
    "The 'dendro' in 'dendrogram' comes from the Greek word 'dendron', meaning 'tree'.",
    "Hierarchical clustering is used in various fields, including biology for phylogenetic tree construction.",
    "The time complexity of naive hierarchical clustering algorithms is O(n^3), making them challenging for large datasets.",
    "Hierarchical clustering doesn't require specifying the number of clusters in advance, unlike k-means clustering.",
    "The linkage method in hierarchical clustering determines how the distance between clusters is calculated.",
    "Sentence transformers use advanced neural networks to convert sentences into dense vector representations.",
]
st.info(f"Did you know? {np.random.choice(facts)}")