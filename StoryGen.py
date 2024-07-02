import streamlit as st
import openai
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import random
import time

# Set up OpenAI API (you'll need to provide your own API key)
openai.api_key = "sk-proj-zyHJL9X0HdAHyjpCg354T3BlbkFJHGLmQ6uwNuHTMQByq0um"

st.set_page_config(layout="wide", page_title="Multiverse Story Generator")

st.title("🌌 Multiverse Story Generator")

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp > header {
        background-color: #1E1E1E;
    }
    .main > div {
        padding-top: 2rem;
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #BB86FC;
    }
    .stButton>button {
        background-color: #03DAC6;
        color: #000000;
    }
    .stTextInput>div>div>input {
        background-color: #3E3E3E;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'story_graph' not in st.session_state:
    st.session_state.story_graph = nx.DiGraph()
    st.session_state.current_node = 0
    st.session_state.story = []

# Function to generate story continuation
def generate_continuation(prompt, choices=2):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=choices,
        stop=None,
        temperature=0.7,
    )
    return [choice.text.strip() for choice in response.choices]

# Function to visualize the story graph
def visualize_graph():
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    
    for node in st.session_state.story_graph.nodes(data=True):
        net.add_node(node[0], label=node[1]['text'][:20] + "...", title=node[1]['text'])
    
    for edge in st.session_state.story_graph.edges():
        net.add_edge(edge[0], edge[1])
    
    net.toggle_physics(False)
    net.show("story_graph.html")
    
    return net.html

# Main story generation loop
if st.button("Start New Story"):
    st.session_state.story_graph.clear()
    st.session_state.current_node = 0
    st.session_state.story = []
    initial_prompt = "Begin an exciting adventure story:"
    continuation = generate_continuation(initial_prompt, choices=1)[0]
    st.session_state.story_graph.add_node(0, text=continuation)
    st.session_state.story.append(continuation)

if st.session_state.story_graph:
    st.markdown(f"### Current story:")
    for i, part in enumerate(st.session_state.story):
        st.markdown(f"**{i+1}.** {part}")
    
    st.markdown("### What happens next?")
    current_text = st.session_state.story_graph.nodes[st.session_state.current_node]['text']
    continuations = generate_continuation(current_text + "\nWhat happens next?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(continuations[0]):
            new_node = len(st.session_state.story_graph)
            st.session_state.story_graph.add_node(new_node, text=continuations[0])
            st.session_state.story_graph.add_edge(st.session_state.current_node, new_node)
            st.session_state.current_node = new_node
            st.session_state.story.append(continuations[0])
    
    with col2:
        if st.button(continuations[1]):
            new_node = len(st.session_state.story_graph)
            st.session_state.story_graph.add_node(new_node, text=continuations[1])
            st.session_state.story_graph.add_edge(st.session_state.current_node, new_node)
            st.session_state.current_node = new_node
            st.session_state.story.append(continuations[1])
    
    st.markdown("### Story Multiverse Visualization")
    st.components.v1.html(visualize_graph(), height=600)

    if st.button("Explore Alternative Timeline"):
        alternative_node = random.choice(list(st.session_state.story_graph.nodes))
        st.session_state.current_node = alternative_node
        st.session_state.story = []
        path = nx.shortest_path(st.session_state.story_graph, source=0, target=alternative_node)
        for node in path:
            st.session_state.story.append(st.session_state.story_graph.nodes[node]['text'])
        st.experimental_rerun()

st.markdown("""
    ## About this App
    This Multiverse Story Generator uses AI to create branching narratives, allowing you to explore different story paths. 
    The visualization shows the structure of the story multiverse you're creating. 
    Each node represents a story fragment, and edges show the connections between them.
    You can start a new story, choose between continuation options, or explore alternative timelines!
""")