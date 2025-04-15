import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re

# Load embedding model (lightweight & fast)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load metadata from JSON file
with open('metadata4_old.txt', 'r') as file:
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

# Modified function to generate step-by-step instructions instead of key points
# Modified function to generate step-by-step instructions instead of key points
def summarize_transcript(video):
    """
    Generate step-by-step instructions from video transcript and metadata.
    """
    transcript_text = video.get("transcript", {}).get("full_text", "")
    timestamped_transcript = video.get("transcript", {}).get("timestamped", [])
    title = video.get("title", "")
    category = video.get("category", "")
    key_features = video.get("key_features_demonstrated", [])
    
    # Initialize steps list
    steps = []
    
    # Determine the type of video to generate appropriate steps
    is_overview = "overview" in title.lower() or "introduction" in category.lower() 
    is_search = "search" in title.lower() or any(k for k in key_features if "search" in k.lower())
    is_filtering = "filter" in title.lower() or any(k for k in key_features if "filter" in k.lower())
    is_analysis = "analysis" in title.lower() or "analytics" in title.lower() or "insights" in title.lower() or "insight studio" in title.lower() or "custom" in title.lower()
    is_dashboard = "dashboard" in title.lower() or any(k for k in key_features if "dashboard" in k.lower())
    
    # For custom analysis videos - prioritizing this check to ensure it's identified correctly
    if "custom analysis" in title.lower() or "insights studio" in title.lower() or (is_analysis and not is_dashboard and not is_filtering):
        steps.append("Step 1: Access Insights Studio by clicking on it in the main navigation menu.")
        steps.append("Step 2: Select 'Custom Analysis' to begin creating your tailored report.")
        
        if "metrics" in transcript_text.lower():
            steps.append("Step 3: Choose from the available measurement categories (Spend Metrics, Membership Metrics, or Utilization Metrics).")
        
        if "filter" in transcript_text.lower() or "population" in transcript_text.lower():
            steps.append("Step 4: Define your target population using the filtering options, selecting criteria such as demographics, diagnosis, or medications.")
        
        if "period" in transcript_text.lower() or "date" in transcript_text.lower():
            steps.append("Step 5: Set your analysis period by selecting predefined options or creating a custom date range.")
            steps.append("Step 6: Choose between pay date, incur date, or incur date with runoff based on your analysis needs.")
        
        if "granularity" in transcript_text.lower():
            steps.append("Step 7: Select granularity options to determine how your data will be grouped (by time period, organizational dimensions, demographics, etc.).")
        
        steps.append("Step 8: Generate your custom report and review the results in the tabular format provided.")
        
        if "export" in transcript_text.lower():
            steps.append("Step 9: Export your completed analysis in your preferred format for sharing with stakeholders.")
    
    # For overview videos
    elif is_overview:
        # [existing overview code remains the same]
        steps.append("Step 1: Access the SIE platform through your authorized login credentials.")
        
        if "components" in transcript_text.lower():
            steps.append("Step 2: Familiarize yourself with the three main components: Automated Dashboards, Insights Studio, and Data Factory.")
            
        if "dashboard" in transcript_text.lower():
            steps.append("Step 3: Explore the Automated Dashboards section to view pre-built reports on standard metrics, wellness programs, and cost management.")
            
        if "search" in transcript_text.lower():
            steps.append("Step 4: Use the search functionality at the top of the interface to quickly locate specific reports or data elements.")
            
        if "favorite" in transcript_text.lower():
            steps.append("Step 5: Mark frequently-used items as favorites by clicking the star icon next to them for quick future access.")
    
    # For search-focused videos
    elif is_search:
        # [existing search code remains the same]
        steps.append("Step 1: Locate the search bar at the top of the SIE platform interface.")
        steps.append("Step 2: Enter relevant keywords related to the content you're looking for (e.g., 'Medical' for medical data reports).")
        
        if "category" in transcript_text.lower() and "results" in transcript_text.lower():
            steps.append("Step 3: Review the search results, which are organized by categories such as dashboards, reports, and data elements.")
            steps.append("Step 4: Use the category tabs to filter results and find exactly what you need.")
        
        if "recently visited" in transcript_text.lower():
            steps.append("Step 5: Access your recent searches and activities through the 'Recently Visited' tab for quick navigation to frequently used items.")
    
    # For filtering videos
    elif is_filtering:
        # [existing filtering code remains the same]
        steps.append("Step 1: Navigate to the Standard Reports section to access your desired dashboard.")
        
        if "medical" in transcript_text.lower() and "category" in transcript_text.lower():
            steps.append("Step 2: Select the appropriate category (Medical, Enrollment, or Pharmacy) based on the data you want to analyze.")
        
        steps.append("Step 3: Locate the filter icon (funnel shape) at the top right corner of the dashboard.")
        
        if "demographic" in transcript_text.lower():
            steps.append("Step 4: Apply demographic filters like gender or relationship type to focus on specific population segments.")
        
        if "diagnosis" in transcript_text.lower():
            steps.append("Step 5: Further refine your analysis by selecting specific diagnosis groups or health conditions if needed.")
        
        steps.append("Step 6: Review the filtered dashboard data to extract the specific insights you need for your analysis.")
    
    # For dashboard videos
    elif is_dashboard:
        # [existing dashboard code remains the same]
        steps.append("Step 1: Navigate to the Standard Reports section in the SIE platform.")
        steps.append("Step 2: Select the appropriate dashboard category based on your analysis needs.")
        
        if "monthly trend" in transcript_text.lower():
            steps.append("Step 3: Choose the Monthly Trend option to view trend data across time periods.")
        
        if "allowed amount" in transcript_text.lower() or "paid amount" in transcript_text.lower():
            steps.append("Step 4: Select whether to view data based on allowed amount or paid amount according to your requirements.")
        
        if "service date" in transcript_text.lower() or "pay date" in transcript_text.lower():
            steps.append("Step 5: Choose between service date or pay date views depending on your analysis perspective.")
        
        if "information icon" in transcript_text.lower() or "i button" in transcript_text.lower():
            steps.append("Step 6: Click on information icons throughout the dashboard to get contextual explanations of the data being displayed.")
        
        if "filter" in transcript_text.lower():
            steps.append("Step 7: Use the filter icon in the top right corner to customize the dashboard view based on specific criteria.")
    
    # [rest of the function remains the same]
    # Generic steps if we couldn't determine specific type
    else:
        # [existing generic code remains the same]
        # Extract action verbs and objects from transcript to create steps
        potential_actions = []
        for line in transcript_text.split("."):
            if "click" in line.lower() or "select" in line.lower() or "choose" in line.lower() or "navigate" in line.lower():
                potential_actions.append(line.strip())
        
        # Turn potential actions into steps
        step_num = 1
        for action in potential_actions[:5]:  # Limit to first 5 actions
            if action:
                # Clean up and format as a step
                action = action.strip()
                # Capitalize first letter and ensure it ends with a period
                if action and not action.endswith('.'):
                    action += '.'
                if action:
                    action = action[0].upper() + action[1:]
                    steps.append(f"Step {step_num}: {action}")
                    step_num += 1
        
        # If we still don't have enough steps, add generic ones based on key features
        if len(steps) < 3 and key_features:
            for feature in key_features[:3]:
                steps.append(f"Step {step_num}: Utilize the {feature} functionality to enhance your data analysis.")
                step_num += 1
    
    # Ensure we have between 3-7 steps
    steps = list(dict.fromkeys(steps))  # Remove any duplicates while preserving order
    
    # If we have too few steps, add generic ones
    if len(steps) < 3:
        platform_name = "SIE platform" if "SIE" in title else "platform"
        generic_steps = [
            f"Step {len(steps)+1}: Access the {platform_name} through your secure login credentials.",
            f"Step {len(steps)+2}: Navigate to the appropriate section based on your analysis needs.",
            f"Step {len(steps)+3}: Use the available filters and options to customize your view.",
            f"Step {len(steps)+4}: Review the generated insights and export if needed."
        ]
        steps.extend(generic_steps[:4-len(steps)])
    
    # Limit to 7 steps maximum to avoid overwhelming the user
    steps = steps[:7]
    
    return steps
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

                # -- Embed YouTube via iframe --
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

                # --- Show transcript summary as step-by-step instructions ---
                steps = summarize_transcript(video)
                st.markdown("**📝 Follow these steps:**")
                for step in steps:
                    st.markdown(f"- {step}")
    else:
        st.error("🤔 Hmm, I couldn't find a relevant video for that query. Please try rephrasing your question!")