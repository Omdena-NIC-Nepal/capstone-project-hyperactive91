import os
import logging
import streamlit as st
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import nltk

# Setup local NLTK data folder
NLTK_DATA_DIR = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Download necessary NLTK data if missing
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=NLTK_DATA_DIR)
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', download_dir=NLTK_DATA_DIR)

download_nltk_resources()

# Initialize logging and VADER analyzer
logging.basicConfig(level=logging.INFO)
vader_analyzer = SentimentIntensityAnalyzer()

ARTICLE_DIR = os.path.join("data", "articles")

def get_article_files(article_dir):
    try:
        return [f for f in os.listdir(article_dir) if f.endswith(('.txt', '.csv', '.json'))]
    except Exception as e:
        logging.error(f"Failed to read article folder: {e}")
        return []

def analyze_text(text):
    # TextBlob analysis
    blob_sentiment = TextBlob(text).sentiment
    # VADER analysis
    vader_scores = vader_analyzer.polarity_scores(text)
    return {
        "textblob": {
            "polarity": blob_sentiment.polarity,
            "subjectivity": blob_sentiment.subjectivity
        },
        "vader": vader_scores
    }

def analyze_sentences(text):
    # Sentence tokenize using official punkt tokenizer ONLY
    sentences = sent_tokenize(text)
    results = []
    for s in sentences:
        scores = vader_analyzer.polarity_scores(s)
        results.append({
            "sentence": s,
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "compound": scores["compound"]
        })
    return results

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
    df.insert(0, "SN", range(1, len(df) + 1))
    st.table(df)

def render_sentence_breakdown(results):
    st.markdown("### Sentence-wise VADER Scores")
    df = pd.DataFrame(results)
    st.dataframe(df)

def display_live_input_analysis():
    st.subheader("Analyze Text Input")
    user_text = st.text_area("Enter your text here", height=200)
    if st.button("Analyze"):
        if user_text.strip():
            scores = analyze_text(user_text)
            render_sentiment_table(scores)
            sentence_scores = analyze_sentences(user_text)
            render_sentence_breakdown(sentence_scores)
        else:
            st.warning("Please enter some text.")

def display_preloaded_sentiment():
    st.subheader("Analyze Article Files")
    files = get_article_files(ARTICLE_DIR)
    if not files:
        st.warning("No article files found.")
        return
    selected = st.selectbox("Select a file", ["Select..."] + files)
    if selected != "Select...":
        path = os.path.join(ARTICLE_DIR, selected)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                scores = analyze_text(content)
                st.markdown(f"Analyzing file: `{selected}`")
                render_sentiment_table(scores)
                sentence_scores = analyze_sentences(content)
                render_sentence_breakdown(sentence_scores)
        except Exception as e:
            st.error(f"Failed to read file `{selected}`: {e}")
    else:
        st.info("Please select a file.")

def main():
    st.title("NLP Sentiment Analyzer")
    st.subheader("Select Input Method")
    mode = st.radio("Choose input method:", ["User Input", "Preloaded Files"])
    if mode == "User Input":
        display_live_input_analysis()
    else:
        display_preloaded_sentiment()

if __name__ == "__main__":
    main()
