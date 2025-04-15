import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re

# Load embedding model (lightweight & fast)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load metadata from JSON file
with open('metadata4.txt', 'r') as file:
    metadata = json.load(file)

# Video metadata
video_metadata = metadata['video_metadata']

# Create a function to get searchable text from a video
def get_searchable_text(video):
    # Get basic info
    title = video.get("title", "")
    description = video.get("description", "")
    keywords = " ".join(video.get("keywords", []))
    
    # Get search aliases if available
    search_aliases = " ".join(video.get("search_aliases", []))
    
    # Get transcript if available
    transcript_text = ""
    if "transcript" in video and isinstance(video["transcript"], dict) and "full_text" in video["transcript"]:
        transcript_text = video["transcript"]["full_text"]
    
    # Combine all text - giving more weight to keywords by repeating them
    combined_text = f"{title} {title} {description} {keywords} {keywords} {keywords} {search_aliases} {transcript_text}"
    
    # Clean up text (remove extra spaces)
    combined_text = re.sub(r'\s+', ' ', combined_text).strip()
    
    return combined_text

# Precompute embeddings for each video
for video in video_metadata:
    searchable_text = get_searchable_text(video)
    video["embedding"] = model.encode(searchable_text)
    
    # Also store the individual keyword embeddings for direct matching
    video["keyword_embeddings"] = []
    if "keywords" in video:
        for keyword in video["keywords"]:
            video["keyword_embeddings"].append({
                "keyword": keyword,
                "embedding": model.encode(keyword)
            })

# Function to find the best matching video with improved logic
def find_best_videos(query, max_results=3):
    # Clean query
    query = query.strip().lower()
    
    # Use a dictionary to track the best score for each video ID to prevent duplicates
    video_scores = {}
    
    # Check for direct keyword matches first (case insensitive)
    for video in video_metadata:
        video_id = video["id"]
        
        # Check title
        if query in video.get("title", "").lower():
            video_scores[video_id] = (video, 0.95)
            continue
            
        # Check keywords
        for keyword in video.get("keywords", []):
            if query in keyword.lower() or keyword.lower() in query:
                video_scores[video_id] = (video, 0.9)
                break
                
        # Check search aliases if available
        if video_id not in video_scores:  # Only check if we haven't matched yet
            for alias in video.get("search_aliases", []):
                if query in alias.lower() or alias.lower() in query:
                    video_scores[video_id] = (video, 0.85)
                    break
    
    # If we have direct matches, extract them
    if video_scores:
        # Get the list of (video, score) tuples
        results = list(video_scores.values())
        # Sort by confidence score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    # If no direct matches, use semantic search
    query_embedding = model.encode(query)
    
    # Calculate similarities with the main embedding
    for video in video_metadata:
        video_id = video["id"]
        main_similarity = cosine_similarity([query_embedding], [video["embedding"]])[0][0]
        
        # Also check individual keyword similarities
        keyword_similarities = []
        for keyword_data in video.get("keyword_embeddings", []):
            keyword_sim = cosine_similarity([query_embedding], [keyword_data["embedding"]])[0][0]
            keyword_similarities.append(keyword_sim)
        
        # Use the max of main similarity and best keyword similarity
        best_keyword_sim = max(keyword_similarities) if keyword_similarities else 0
        final_similarity = max(main_similarity, best_keyword_sim)
        
        # Only store if above threshold
        if final_similarity > 0.4:
            video_scores[video_id] = (video, final_similarity)
    
    # Extract and sort the results
    results = list(video_scores.values())
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:max_results]

# --- Streamlit UI ---

# Page configuration
st.set_page_config(page_title="TRA SmartHelp", layout="centered")

# Logo
st.image("logo.png", width=200)

# Header bar with company style
st.markdown("<h1 style='color:#d32f2f; font-family: Arial, sans-serif;'>TRA SmartHelp</h1>", unsafe_allow_html=True)

# Subheading with gradient background
st.markdown("""
    <div style='background: linear-gradient(90deg, #ff6f61, #ffb74d, #64b5f6); padding: 10px; border-radius: 10px;'>
        <h2 style='color: white; font-family: Arial, sans-serif;'>Curious about the TRA platform? I'm all ears!</h2>
    </div>
    """, unsafe_allow_html=True)

# Chat input
query = st.text_input("💬 Ask me anything!", placeholder="Type your question here...")

if query:
    results = find_best_videos(query)
    
    if results:
        st.success(f"Found {len(results)} relevant video{'s' if len(results) > 1 else ''} for you!")
        
        for i, (video, score) in enumerate(results):
            with st.expander(f"Video {i+1}: {video['title']}", expanded=(i==0)):
                st.markdown(f"**Description**: {video['description']}")
                st.video(video["video_url"])
    else:
        st.error("🤔 Hmm, I couldn't find a relevant video for that query. Please try rephrasing your question!")