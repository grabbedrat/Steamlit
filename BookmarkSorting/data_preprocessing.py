import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from tagging import tag_bookmarks

@st.cache_data
def load_file_content(file):
    return file.read().decode('utf-8')

def extract_bookmarks(_soup):
    bookmarks = []
    for a in _soup.find_all('a'):
        bookmark = {
            'title': a.string,
            'url': a.get('href'),
            'add_date': a.get('add_date'),
            'tags': a.get('tags', '')
        }
        bookmarks.append(bookmark)
    return pd.DataFrame(bookmarks).rename(columns=lambda x: x.lower())

@st.cache_data
def preprocess_data(_bookmarks_df, remove_duplicates, duplicate_criterion, trim_titles, max_title_length, generate_tags, batch_size):
    original_count = len(_bookmarks_df)

    if remove_duplicates:
        _bookmarks_df = _bookmarks_df.drop_duplicates(subset=[duplicate_criterion], keep='first')

    if trim_titles:
        _bookmarks_df['title'] = _bookmarks_df['title'].apply(lambda x: x[:max_title_length] if x else x)

    intermediate_count = len(_bookmarks_df)

    if generate_tags:
        _bookmarks_df, tagging_info = tag_bookmarks(_bookmarks_df, batch_size)
    else:
        tagging_info = {"tagged_count": 0, "error_count": 0, "total_batches": 0, "processed_batches": 0}

    final_count = len(_bookmarks_df)
    
    return _bookmarks_df, tagging_info, original_count, intermediate_count, final_count

def load_and_preprocess_data(uploaded_file):
    content = load_file_content(uploaded_file)
    soup = BeautifulSoup(content, 'html.parser')
    bookmarks_df = extract_bookmarks(soup)
    
    # Data Preprocessing
    st.header("Data Preprocessing")
    with st.expander("Preprocessing Options", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            remove_duplicates = st.checkbox("Remove Duplicates", value=True, key="remove_duplicates")
            duplicate_criterion = st.radio("Duplicate Criterion", ["url", "title"], key="duplicate_criterion")
        with col2:
            trim_titles = st.checkbox("Trim Long Titles", value=True, key="trim_titles")
            max_title_length = st.number_input("Max Title Length", min_value=10, max_value=200, value=100, key="max_title_length")
        with col3:
            generate_tags = st.checkbox("Generate Tags", value=True, key="generate_tags")
            batch_size = st.number_input("Batch Size", min_value=1, max_value=50, value=10, key="batch_size")

    bookmarks_df, tagging_info, original_count, intermediate_count, final_count = preprocess_data(
        bookmarks_df, remove_duplicates, duplicate_criterion, trim_titles, max_title_length, generate_tags, batch_size
    )

    st.write(f"Bookmarks reduced from {original_count} to {intermediate_count} after duplicate removal and title trimming.")
    
    if generate_tags:
        st.write(f"Tagging complete. {tagging_info['tagged_count']} bookmarks tagged in {tagging_info['processed_batches']} batches.")
        st.write(f"Total batches: {tagging_info['total_batches']}, Batch size: {batch_size}")
        if tagging_info['error_count'] > 0:
            st.warning(f"{tagging_info['error_count']} errors occurred during tagging.")

    st.write(f"Final number of bookmarks after all preprocessing: {final_count}")
    
    return bookmarks_df, tagging_info