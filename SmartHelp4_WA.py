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
    title = video.get("title", "")
    description = video.get("description", "")
    keywords = " ".join(video.get("keywords", []))
    search_aliases = " ".join(video.get("search_aliases", []))
    transcript_text = ""
    if "transcript" in video and isinstance(video["transcript"], dict) and "full_text" in video["transcript"]:
        transcript_text = video["transcript"]["full_text"]
    combined_text = f"{title} {title} {description} {keywords} {keywords} {keywords} {search_aliases} {transcript_text}"
    combined_text = re.sub(r'\s+', ' ', combined_text).strip()
    return combined_text

# Precompute embeddings for each video
for video in video_metadata:
    searchable_text = get_searchable_text(video)
    video["embedding"] = model.encode(searchable_text)
    video["keyword_embeddings"] = []
    if "keywords" in video:
        for keyword in video["keywords"]:
            video["keyword_embeddings"].append({
                "keyword": keyword,
                "embedding": model.encode(keyword)
            })

# Function to find the best matching videos
def find_best_videos(query, max_results=3):
    query = query.strip().lower()
    video_scores = {}

    # Direct match search
    for video in video_metadata:
        video_id = video["id"]
        if query in video.get("title", "").lower():
            video_scores[video_id] = (video, 0.95)
            continue
        for keyword in video.get("keywords", []):
            if query in keyword.lower() or keyword.lower() in query:
                video_scores[video_id] = (video, 0.9)
                break
        if video_id not in video_scores:
            for alias in video.get("search_aliases", []):
                if query in alias.lower() or alias.lower() in query:
                    video_scores[video_id] = (video, 0.85)
                    break

    if video_scores:
        results = list(video_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    # Semantic match
    query_embedding = model.encode(query)
    for video in video_metadata:
        video_id = video["id"]
        main_similarity = cosine_similarity([query_embedding], [video["embedding"]])[0][0]
        keyword_similarities = []
        for keyword_data in video.get("keyword_embeddings", []):
            keyword_sim = cosine_similarity([query_embedding], [keyword_data["embedding"]])[0][0]
            keyword_similarities.append(keyword_sim)
        best_keyword_sim = max(keyword_similarities) if keyword_similarities else 0
        final_similarity = max(main_similarity, best_keyword_sim)
        if final_similarity > 0.4:
            video_scores[video_id] = (video, final_similarity)

    results = list(video_scores.values())
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:max_results]

# --- Streamlit UI ---

st.set_page_config(page_title="TRA SmartHelp", layout="centered")
st.image("logo.png", width=200)

st.markdown("<h1 style='color:#d32f2f; font-family: Arial, sans-serif;'>TRA SmartHelp</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='background: linear-gradient(90deg, #ff6f61, #ffb74d, #64b5f6); padding: 10px; border-radius: 10px;'>
        <h2 style='color: white; font-family: Arial, sans-serif;'>Curious about the TRA platform? I'm all ears!</h2>
    </div>
    """, unsafe_allow_html=True)

query = st.text_input("💬 Ask me anything!", placeholder="Type your question here...")

if query:
    results = find_best_videos(query)
    if results:
        st.success(f"Found {len(results)} relevant video{'s' if len(results) > 1 else ''} for you!")
        for i, (video, score) in enumerate(results):
            with st.expander(f"Video {i+1}: {video['title']}", expanded=(i == 0)):
                st.markdown(f"**Description**: {video['description']}")

                # -- Embed YouTube via iframe with rel=0 to avoid suggestions --
                video_url = video["video_url"]
                if "youtube.com" in video_url and "v=" in video_url:
                    video_id = video_url.split("v=")[-1].split("&")[0]
                elif "youtu.be" in video_url:
                    video_id = video_url.split("/")[-1]
                else:
                    video_id = ""

                if video_id:
                    iframe_html = f"""
                    <iframe width="100%" height="315"
                    src="https://www.youtube.com/embed/{video_id}?rel=0&modestbranding=1&showinfo=0"
                    frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
                    allowfullscreen></iframe>
                    """
                    st.markdown(iframe_html, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Could not embed video. Invalid YouTube URL.")
    else:
        st.error("🤔 Hmm, I couldn't find a relevant video for that query. Please try rephrasing your question!")
