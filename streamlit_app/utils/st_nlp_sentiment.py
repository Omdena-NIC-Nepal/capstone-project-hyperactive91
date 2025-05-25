import os
import logging
import streamlit as st
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sent_tokenize
import nltk

# Safe NLTK downloads
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize
logging.basicConfig(level=logging.INFO)
vader_analyzer = SentimentIntensityAnalyzer()

# --- CONFIG ---
ARTICLE_DIR = os.path.join("data", "articles")

# Load article filenames
def get_article_files(article_dir):
    try:
        return [f for f in os.listdir(article_dir) if f.endswith(('.txt', '.csv', '.json'))]
    except Exception as e:
        logging.error(f"Failed to read article folder: {e}")
        return []

# Analyze with TextBlob + VADER
def analyze_text(text):
    blob = TextBlob(text).sentiment
    vader = vader_analyzer.polarity_scores(text)

    return {
        "textblob": {
            "polarity": blob.polarity,
            "subjectivity": blob.subjectivity
        },
        "vader": vader
    }

# Sentence-wise VADER
def analyze_by_sentence(text):
    sentences = sent_tokenize(text)
    results = []
    for sent in sentences:
        scores = vader_analyzer.polarity_scores(sent)
        results.append({
            "sentence": sent,
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "compound": scores["compound"]
        })
    return results

# Display overall scores
def render_sentiment_table(scores):
    st.markdown("### Overall Sentiment Scores")
    data = {
        "Metric": [
            "TextBlob Polarity", "TextBlob Subjectivity",
            "VADER Negative", "VADER Neutral", "VADER Positive", "VADER Compound"
        ],
        "Score": [
            round(scores['textblob']['polarity'], 2),
            round(scores['textblob']['subjectivity'], 2),
            round(scores['vader']['neg'], 2),
            round(scores['vader']['neu'], 2),
            round(scores['vader']['pos'], 2),
            round(scores['vader']['compound'], 2)
        ]
    }
    df = pd.DataFrame(data)
    df.insert(0, "SN", range(1, len(df)+1))
    st.table(df)

# Display sentence-wise breakdown
def render_sentence_breakdown(results):
    st.markdown("### Sentence-wise VADER Scores")
    df = pd.DataFrame(results)
    st.dataframe(df)

# Analyze user input
def display_live_input_analysis():
    st.subheader("Analyze Text Input")
    article_text = st.text_area("Enter your text below", height=200)

    if st.button("Analyze"):
        if article_text.strip():
            scores = analyze_text(article_text)
            render_sentiment_table(scores)

            sentence_scores = analyze_by_sentence(article_text)
            render_sentence_breakdown(sentence_scores)
        else:
            st.warning("Please enter some text.")

# Analyze from preloaded files
def display_preloaded_sentiment():
    st.subheader("Analyze Article Files")
    files = get_article_files(ARTICLE_DIR)

    if not files:
        st.warning("No article files found.")
        return

    selected = st.selectbox("Choose a file:", ["Select..."] + files)
    if selected != "Select...":
        path = os.path.join(ARTICLE_DIR, selected)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                scores = analyze_text(content)
                st.markdown(f"Analyzing: `{selected}`")
                render_sentiment_table(scores)

                sentence_scores = analyze_by_sentence(content)
                render_sentence_breakdown(sentence_scores)
        except Exception as e:
            st.error(f"Failed to read file `{selected}`: {e}")
    else:
        st.info("Please select a file.")

# Main app
def main():
    st.title("NLP Sentiment Analyzer")
    st.subheader("Your input choice : ")
    mode = st.radio("Choose an option:", ["User Input", "Saved Files"])

    if mode == "User Input":
        display_live_input_analysis()
    else:
        display_preloaded_sentiment()

if __name__ == "__main__":
    main()
