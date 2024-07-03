import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import re
import numpy as np

@st.cache_data
def generate_embeddings(titles, urls, tags):
    # Extract text from bookmarks (title and URL)
    texts = [f"{title} {url}" for title, url in zip(titles, urls)]
    
    # Clean texts
    texts = [re.sub(r'http\S+', '', text) for text in texts]
    texts = [re.sub(r'\s+', ' ', text).strip() for text in texts]
    
    # Generate embeddings using TF-IDF for texts
    text_vectorizer = TfidfVectorizer()
    text_embeddings = text_vectorizer.fit_transform(texts).toarray()
    
    # Generate embeddings for tags
    tag_vectorizer = TfidfVectorizer()
    tag_embeddings = tag_vectorizer.fit_transform(tags).toarray()
    
    # Inside the generate_embeddings function
    text_weight = 0.5
    tag_weight = 0.5
    combined_embeddings = np.hstack((text_embeddings * text_weight, tag_embeddings * tag_weight))
    
    # Normalize the combined embeddings
    normalized_embeddings = normalize(combined_embeddings)
    
    return normalized_embeddings