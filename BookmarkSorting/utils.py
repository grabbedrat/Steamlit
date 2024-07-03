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