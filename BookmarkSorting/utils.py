from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def generate_prompts(labels, bookmarks_df):
    all_prompts = []
    
    def generate_prompt(cluster_id, indent=""):
        folder_name = f"Folder {cluster_id}"
        bookmarks_in_cluster = bookmarks_df[labels == cluster_id]
        
        if not bookmarks_in_cluster.empty:
            bookmark_titles = bookmarks_in_cluster['title'].tolist()
            bookmark_titles_str = "\n".join([f"{indent}  - {title}" for title in bookmark_titles])
            prompt = f"{indent}Folder: {folder_name}\n{bookmark_titles_str}"
            return prompt
        return ""

    for cluster in sorted(set(labels)):
        if cluster != -1:  # Exclude noise points
            prompt = generate_prompt(cluster)
            if prompt:
                all_prompts.append(prompt)

    return all_prompts

def perform_lsa(bookmarks_df, n_components):
    # Combine title, url, and tags into a single text field
    bookmarks_df['text'] = bookmarks_df['title'] + ' ' + bookmarks_df['url'] + ' ' + bookmarks_df['tags']
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(bookmarks_df['text'])
    
    # Perform LSA using TruncatedSVD
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    
    return lsa_matrix