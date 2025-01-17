import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.cluster.hierarchy import fcluster
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def generate_llm_naming_prompt(folder_name, subfolders):
    prompt = f"Please provide a descriptive name for the following bookmark folder based on its contents:\n\n{folder_name}:\n"
    
    if isinstance(subfolders, dict):
        for low_level, items in subfolders.items():
            prompt += f"  {low_level}:\n"
            for title, _ in items[:5]:  # List up to 5 bookmark titles for each subfolder
                prompt += f"    - {title}\n"
            if len(items) > 5:
                prompt += f"    - ... ({len(items) - 5} more)\n"
    else:
        for title, _ in subfolders[:5]:  # List up to 5 bookmark titles for the folder
            prompt += f"  - {title}\n"
        if len(subfolders) > 5:
            prompt += f"  - ... ({len(subfolders) - 5} more)\n"
    
    prompt += "\nPlease provide a concise, descriptive name for this folder that captures its overall theme or content. Please return in the format 'Current Name: New Name'.\n\n"
    return prompt

def generate_prompts(labels, bookmarks_df, linkage_matrix):
    n_clusters = min(10, len(set(labels)) - 1)  # Adjust the number of clusters, excluding noise
    
    if n_clusters > 1:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        label_mapping = {original: hierarchical for original, hierarchical in zip(range(len(cluster_labels)), cluster_labels)}
        hierarchy = defaultdict(lambda: defaultdict(list))
        for label, title, url in zip(labels, bookmarks_df['title'], bookmarks_df['url']):
            if label != -1:
                high_level = f"Folder {label_mapping.get(label, 'Misc')}"
                low_level = f"Subfolder {label}"
                hierarchy[high_level][low_level].append((title, url))
            else:
                hierarchy['Uncategorized']['Noise'].append((title, url))
    else:
        hierarchy = defaultdict(list)
        for label, title, url in zip(labels, bookmarks_df['title'], bookmarks_df['url']):
            if label != -1:
                hierarchy[f"Folder {label}"].append((title, url))
            else:
                hierarchy['Uncategorized'].append((title, url))

    # Generate LLM naming prompts for each folder and subfolder
    llm_prompts = {}
    for folder_name, subfolders in hierarchy.items():
        folder_content = []
        if isinstance(subfolders, dict):
            subfolder_prompts = {}
            for subfolder_name, items in subfolders.items():
                subfolder_content = [title for title, _ in items[:5]]
                if len(items) > 5:
                    subfolder_content.append(f"... ({len(items) - 5} more)")
                subfolder_prompts[subfolder_name] = {
                    "type": "subfolder",
                    "parent": folder_name,
                    "content": subfolder_content
                }
                folder_content.append(f"{subfolder_name}: {', '.join(subfolder_content)}")
            llm_prompts[folder_name] = {
                "type": "folder",
                "content": folder_content,
                "subfolders": subfolder_prompts
            }
        else:
            folder_content = [title for title, _ in subfolders[:5]]
            if len(subfolders) > 5:
                folder_content.append(f"... ({len(subfolders) - 5} more)")
            llm_prompts[folder_name] = {
                "type": "folder",
                "content": folder_content
            }
    
    return llm_prompts, hierarchy

def generate_html_structure(hierarchy, current_timestamp):
    all_prompts = [
        '<!DOCTYPE NETSCAPE-Bookmark-file-1>',
        '<!-- This is an automatically generated file.',
        ' It will be read and overwritten.',
        ' DO NOT EDIT! -->',
        '<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">',
        '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'; script-src \'none\'; img-src data: *; object-src \'none\'"></meta>',
        '<TITLE>Bookmarks</TITLE>',
        '<H1>Bookmarks Menu</H1>',
        '<DL><p>',
        f'<DT><H3 ADD_DATE="{current_timestamp}" LAST_MODIFIED="{current_timestamp}" PERSONAL_TOOLBAR_FOLDER="true">Bookmarks Toolbar</H3>',
        '<DL><p>'
    ]

    for high_level, subfolders in hierarchy.items():
        all_prompts.append(f'<DT><H3 ADD_DATE="{current_timestamp}" LAST_MODIFIED="{current_timestamp}">{high_level}</H3>')
        all_prompts.append('<DL><p>')
        if isinstance(subfolders, dict):
            for low_level, items in subfolders.items():
                all_prompts.append(f'<DT><H3 ADD_DATE="{current_timestamp}" LAST_MODIFIED="{current_timestamp}">{low_level}</H3>')
                all_prompts.append('<DL><p>')
                for title, url in items:
                    all_prompts.append(f'<DT><A HREF="{url}" ADD_DATE="{current_timestamp}" LAST_MODIFIED="{current_timestamp}">{title}</A>')
                all_prompts.append('</DL><p>')
        else:
            for title, url in subfolders:
                all_prompts.append(f'<DT><A HREF="{url}" ADD_DATE="{current_timestamp}" LAST_MODIFIED="{current_timestamp}">{title}</A>')
        all_prompts.append('</DL><p>')

    all_prompts.append('</DL><p>')
    
    return "\n".join(all_prompts)

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