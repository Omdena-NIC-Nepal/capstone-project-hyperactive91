import os
import logging
import streamlit as st
from textblob import TextBlob
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION ---
ARTICLE_DIR = os.path.join("data", "articles")

# Load article files from folder
def get_article_files(article_dir):
    try:
        files = [f for f in os.listdir(article_dir) if f.endswith(('.txt', '.csv', '.json'))]
        return files
    except Exception as e:
        logging.error(f"Failed to read article folder: {e}")
        return []

# Extract sentiment scores
def extract_sentiment_scores(model_dict):
    try:
        if isinstance(model_dict, dict) and 'textblob' in model_dict and 'vader' in model_dict:
            textblob = model_dict.get("textblob", {})
            vader = model_dict.get("vader", {})
            return {
                "textblob": {
                    "polarity": textblob.get("polarity", 0.0),
                    "subjectivity": textblob.get("subjectivity", 0.0)
                },
                "vader": {
                    "neg": vader.get("neg", 0.0),
                    "neu": vader.get("neu", 1.0),
                    "pos": vader.get("pos", 0.0),
                    "compound": vader.get("compound", 0.0)
                }
            }
    except Exception as e:
        logging.error(f"Error extracting sentiment: {e}")
    return None

# Fallback sentiment analysis using TextBlob
def analyze_with_textblob(text):
    blob = TextBlob(text).sentiment
    return {
        "textblob": {
            "polarity": blob.polarity,
            "subjectivity": blob.subjectivity
        },
        "vader": {
            "neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0
        }
    }

# Render results as table
def render_sentiment_table(scores: dict):
    st.markdown("### Sentiment Scores")

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

    styled_table = df.to_html(index=False, classes='centered-table')

    st.markdown(
        """
        <style>
        .centered-table {
            width: 100%;
            border-collapse: collapse;
        }
        .centered-table th, .centered-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .centered-table th {
            background-color: #f2f2f2;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(styled_table, unsafe_allow_html=True)

# Analyze live input
def display_live_input_analysis():
    st.subheader("Analyze Text Input")
    article_text = st.text_area("Enter text below", height=200)

    if st.button("Analyze"):
        if article_text.strip():
            result = analyze_with_textblob(article_text)
            render_sentiment_table(result)
        else:
            st.warning("Please enter some text.")

# Analyze articles from directory
def display_preloaded_sentiment():
    st.subheader("Analyze Article Files")

    article_files = get_article_files(ARTICLE_DIR)
    if not article_files:
        st.warning("No article files found.")
        return

    selected_file = st.selectbox("Choose a file:", ["Select..."] + article_files)

    if selected_file != "Select...":
        file_path = os.path.join(ARTICLE_DIR, selected_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                article_text = f.read()
                result = analyze_with_textblob(article_text)
                st.markdown(f"Analyzing: `{selected_file}`")
                render_sentiment_table(result)
        except Exception as e:
            st.error(f"Error reading file `{selected_file}`: {e}")
    else:
        st.info("Please select a file.")

# Main app
def main():
    st.title("NLP Sentiment Analyzer")
    st.subheader("Select Input Method")
    mode = st.radio("Choose an option:", ["User Input", "Files"])

    if mode == "User Input":
        display_live_input_analysis()
    else:
        display_preloaded_sentiment()

# Run the app
if __name__ == "__main__":
    main()
