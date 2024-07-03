import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import re

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