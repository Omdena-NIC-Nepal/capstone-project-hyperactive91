import streamlit as st
from summa.summarizer import summarize
import os
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)

# -------------------- Summarization Function --------------------
def summarize_text_with_summa(text, ratio=0.3):
    try:
        if not text.strip():
            return "No input text provided."
        summary = summarize(text, ratio=ratio)
        return summary if summary else "Text too short or unstructured to summarize effectively."
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return f"Error: {e}"

# -------------------- Load Files from Articles Folder --------------------
@st.cache_data
def load_article_texts():
    folder_path = r"C:\Users\Wlink\omdena\Assignment\Final_project\capstone-project-hyperactive91\data\articles"
    files_dict = {}

    if not os.path.exists(folder_path):
        st.error(f"❌ Folder not found: {folder_path}")
        return files_dict

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            try:
                with open(full_path, "r", encoding="utf-8") as file:
                    files_dict[filename] = file.read()
            except Exception as e:
                st.error(f"❌ Could not read {filename}: {e}")
    return files_dict

# -------------------- Render Summary + Stats --------------------
def render_summary_text(original_text, summary_text):
    st.subheader("Summary")
    st.text_area("Summary Output", summary_text, height=200)

    df = pd.DataFrame({
        "Metric": ["Word Count", "Character Count"],
        "Original": [len(original_text.split()), len(original_text)],
        "Summary": [len(summary_text.split()), len(summary_text)],
    })

    df.insert(0, "SN", range(1, len(df) + 1))

    st.markdown(df.to_html(index=False), unsafe_allow_html=True)

# -------------------- Streamlit UI --------------------
def main():
    st.title("NLP Text Summarization")
    option = st.radio("Choose Input Method", ["User Input", "From Files"])

    if option == "User Input":
        user_input = st.text_area("Enter your text here:", height=200)
        if st.button("Summarize"):
            if user_input.strip():
                summary = summarize_text_with_summa(user_input)
                render_summary_text(user_input, summary)
            else:
                st.warning("Please enter some text.")

    else:  # From Files
        articles = load_article_texts()
        if not articles:
            st.warning("⚠️ No .txt files found in the articles folder.")
            return

        filenames = ["Select..."] + list(articles.keys())
        selected = st.selectbox("Choose a file", filenames)

        if selected != "Select...":
            original = articles[selected]
            summary = summarize_text_with_summa(original)
            render_summary_text(original, summary)
        else:
            st.info("ℹ️ Please select a file to summarize.")

# -------------------- Run App --------------------
if __name__ == "__main__":
    main()
