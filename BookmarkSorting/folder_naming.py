import re
import asyncio
import aiohttp
import ssl
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

# Assume OpenAI API key is set in st.secrets["OPENAI_API_KEY"]

# Create a custom SSL context that doesn't verify certificates
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def generate_naming_prompt(name, info):
    prompt = f"Please provide a new, descriptive name for the following {'folder' if info['type'] == 'folder' else 'subfolder'} based on its contents. Always provide a new name, even if the current name seems appropriate:\n\n{name}:\n"
    for item in info['content']:
        prompt += f"  - {item}\n"
    prompt += "\nProvide a concise, descriptive name that captures the overall theme or content. Return ONLY the new name, without any additional text or explanations.\n\n"
    return prompt

def extract_new_name(old_name, llm_response):
    # Try to find a new name in the format "Old Name: New Name"
    match = re.search(f"{re.escape(old_name)}:\s*(.+)", llm_response)
    if match:
        return match.group(1).strip()
    
    # If not found, try to find any colon-separated pair and return the second part
    match = re.search(":\s*(.+)", llm_response)
    if match:
        return match.group(1).strip()
    
    # If still not found, return the entire response as the new name
    return llm_response.strip()

async def generate_names_for_batch(batch, session):
    async def generate_name_for_item(name, info):
        prompt = generate_naming_prompt(name, info)
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "n": 1,
                    "stop": None,
                    "temperature": 0.5,
                },
                ssl=ssl_context
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    llm_response = data['choices'][0]['message']['content'].strip()
                    new_name = extract_new_name(name, llm_response)
                    return name, new_name, info['type'], info.get('parent')
                else:
                    st.error(f"Error generating name: HTTP {response.status}")
                    return name, name, info['type'], info.get('parent')
        except Exception as e:
            st.error(f"Error generating name: {str(e)}")
            return name, name, info['type'], info.get('parent')

    tasks = [generate_name_for_item(name, info) for name, info in batch]
    return await asyncio.gather(*tasks)

async def process_batches(batches):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        results = []
        for batch in batches:
            batch_results = await generate_names_for_batch(batch, session)
            results.extend(batch_results)
        return results

def name_folders(llm_prompts, batch_size=5):
    items_to_name = []
    for folder_name, folder_info in llm_prompts.items():
        items_to_name.append((folder_name, folder_info))
        if 'subfolders' in folder_info:
            for subfolder_name, subfolder_info in folder_info['subfolders'].items():
                items_to_name.append((subfolder_name, subfolder_info))

    total_to_name = len(items_to_name)
    
    if total_to_name == 0:
        return {}

    batches = [items_to_name[i:i + batch_size] for i in range(0, total_to_name, batch_size)]
    total_batches = len(batches)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process batches
    with ThreadPoolExecutor() as executor:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = asyncio.ensure_future(process_batches(batches))
        naming_results = loop.run_until_complete(future)

    # Create dictionary of new folder names
    new_names = {}
    for old_name, new_name, item_type, parent in naming_results:
        if item_type == 'folder':
            new_names[old_name] = {'name': new_name, 'subfolders': {}}
        else:  # subfolder
            parent_folder = next(folder for folder, info in llm_prompts.items() if 'subfolders' in info and old_name in info['subfolders'])
            new_names[parent_folder]['subfolders'][old_name] = new_name

        progress = min(len(naming_results) / total_to_name, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing item {len(naming_results)} of {total_to_name}")

    status_text.text("Folder naming complete!")

    return new_names

def update_folder_names(hierarchy, new_names):
    updated_hierarchy = {}
    for folder_name, subfolders in hierarchy.items():
        new_folder_info = new_names.get(folder_name, {'name': folder_name, 'subfolders': {}})
        new_folder_name = new_folder_info['name']
        if isinstance(subfolders, dict):
            updated_subfolders = {}
            for subfolder_name, items in subfolders.items():
                new_subfolder_name = new_folder_info['subfolders'].get(subfolder_name, subfolder_name)
                updated_subfolders[new_subfolder_name] = items
            updated_hierarchy[new_folder_name] = updated_subfolders
        else:
            updated_hierarchy[new_folder_name] = subfolders
    return updated_hierarchy