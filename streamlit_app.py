from transformers import pipeline
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
from googleapiclient.discovery import build

# ------------------ API Key ------------------

api_key = "AIzaSyDKAniwm6Sfjf7tCfMsiA8CoNM_iL9ipXE"

# ------------------ FUNCTION DEFINITIONS ------------------

def extract_video_id(url):
    match = re.search(r"(v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(2) if match else None


def fetch_comments_from_video(video_id,  max_comments=500):
    api_key = "AIzaSyDKAniwm6Sfjf7tCfMsiA8CoNM_iL9ipXE"
    youtube = build('youtube', 'v3', developerKey=api_key)  # YouTube API client built successfully.

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat='plainText',
        ).execute()  # Sending request to YouTube API

        # items ek list hai jisme youtube api ne response yaani requested comm. diya hai
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

            if len(comments) > max_comments:
                break

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
        time.sleep(1)

    # 💾 Save to CSV
    df = pd.DataFrame(comments, columns=["Comment"])
    df.to_csv("youtube_comments.csv", index=False)
    print('Done ', len(comments), "comments saved to youtube_comments.csv")

classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def is_valid_comment(comment):
    if comment is None:
        return False
    comment = str(comment).strip()
    if comment == "":
        return False
    return bool(re.search(r'[a-zA-Z]', comment))  # At least one letter hona chahiye



# Get sentiment for valid comment (ignore Neutral)
def get_sentiment(comment):
    try:
        result = classifier(str(comment)[:512])[0]
        label = result['label']
        stars = int(label[0])  # '4 stars' -> 4

        if stars >= 4:
            return "Positive"
        elif stars == 3:
            return "Neutral"
        else:
            return "Negative"
    except:
        return "Negative"


def analyze_sentiments(input_file="youtube_comments.csv", output_file="youtube_comments_with_sentiments.csv"):
    df = pd.read_csv(input_file)

    # Filter out invalid comments
    df = df[df['Comment'].apply(is_valid_comment)]

    # Apply sentiment
    df['Sentiment'] = df['Comment'].apply(get_sentiment)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Sentiment Analysis Completed! Saved to '{output_file}'")

# ------------------ STREAMLIT APP ------------------

st.set_page_config("YouTube Sentiment Dashboard", layout="centered", page_icon="📊")
st.title("📊YouTube Comment Sentiment Analyzer")

video_url = st.text_input("🔗 Enter YouTube Video URL")

if st.button("Analyze"):
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("❌ Invalid YouTube Video URL!")
    else:
        with st.spinner("📥 Fetching comments..."):
            fetch_comments_from_video(video_id)

        with st.spinner("🔍 Analyzing Sentiments..."):
            df = analyze_sentiments()

        st.success("✅ Done! Comments fetched & analyzed.")

        df = pd.read_csv("youtube_comments_with_sentiments.csv")
        st.write("### 📝 Sample Comments with Sentiment")
        st.dataframe(df.head(10))

        sentiment_counts = df['Sentiment'].value_counts()

        # Define correct colors for each sentiment
        color_map = {
            'Negative': '#F44336',  # Red
            'Neutral': '#FFEB3B',  # Yellow
            'Positive': '#4CAF50'  # Green
        }

        # Create color list in the same order as sentiment_counts
        colors = [color_map[sentiment] for sentiment in sentiment_counts.index]

        # Plot pie chart
        fig, ax = plt.subplots(figsize=(3,3))
        sentiment_counts.plot.pie(autopct='%1.1f%%', colors=colors, startangle=90, ax=ax)
        ax.set_ylabel('')
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

#-------------------------------Footer-------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Made with ❤ by Saumya</div>", unsafe_allow_html=True
)