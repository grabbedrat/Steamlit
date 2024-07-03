import openai
import streamlit as st
import pandas as pd
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import ssl

# Set your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Create a custom SSL context that doesn't verify certificates
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def generate_tags_for_batch(batch, session):
    async def generate_tags_for_bookmark(title, url):
        prompt = f"Generate 5 relevant, comma-separated tags for the following bookmark:\nTitle: {title}\nURL: {url}\n\nTags:"
        
        try:
            async with session.post(
                "https://api.openai.com/v1/completions",
                headers={"Authorization": f"Bearer {openai.api_key}"},
                json={
                    "model": "gpt-3.5-turbo-instruct",
                    "prompt": prompt,
                    "max_tokens": 50,
                    "n": 1,
                    "stop": None,
                    "temperature": 0.5,
                },
                ssl=ssl_context  # Use the custom SSL context
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    tags = data['choices'][0]['text'].strip()
                    return tags
                else:
                    st.error(f"Error generating tags: HTTP {response.status}")
                    return ""
        except Exception as e:
            st.error(f"Error generating tags: {str(e)}")
            return ""

    tasks = [generate_tags_for_bookmark(bookmark['title'], bookmark['url']) for bookmark in batch]
    return await asyncio.gather(*tasks)

async def process_batches(batches):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        results = []
        for batch in batches:
            batch_results = await generate_tags_for_batch(batch, session)
            results.extend(batch_results)
        return results


def tag_bookmarks(bookmarks_df, batch_size=10):
    bookmarks_to_tag = bookmarks_df[bookmarks_df['tags'].isna() | (bookmarks_df['tags'] == '')].to_dict('records')
    total_to_tag = len(bookmarks_to_tag)
    
    if total_to_tag == 0:
        return bookmarks_df, {"tagged_count": 0, "error_count": 0, "total_batches": 0, "processed_batches": 0}

    batches = [bookmarks_to_tag[i:i + batch_size] for i in range(0, total_to_tag, batch_size)]
    total_batches = len(batches)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process batches
    with ThreadPoolExecutor() as executor:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = asyncio.ensure_future(process_batches(batches))
        tags_results = loop.run_until_complete(future)

    # Update DataFrame with new tags
    tag_index = 0
    error_count = 0
    for i, row in bookmarks_df.iterrows():
        if pd.isna(row['tags']) or row['tags'] == '':
            if tags_results[tag_index] == "":
                error_count += 1
            bookmarks_df.at[i, 'tags'] = tags_results[tag_index]
            tag_index += 1
        
        progress = min(tag_index / total_to_tag, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing bookmark {tag_index} of {total_to_tag}")

    status_text.text("Tagging complete!")
    
    tagging_info = {
        "tagged_count": tag_index,
        "error_count": error_count,
        "total_batches": total_batches,
        "processed_batches": total_batches  # Assuming all batches were processed
    }
    
    return bookmarks_df, tagging_info