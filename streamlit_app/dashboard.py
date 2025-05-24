import streamlit as st
from utils.st_exploratory import show_analysis
from utils.load import load_data
from utils.st_modeltraining import render
from utils.st_modelprediction import predict
from utils import st_nlp_sentiment
from utils import st_nlp_summarize

# Load data once
df = load_data()

# Initialize session state
if "main_section" not in st.session_state:
    st.session_state.main_section = "Select..."
if "sub_page" not in st.session_state:
    st.session_state.sub_page = "Select..."
if "page" not in st.session_state:
    st.session_state.page = "My Dashboard"
if "nlp_section" not in st.session_state:
    st.session_state.nlp_section = "Select..."

# Sidebar Layout
st.sidebar.markdown("### Navigation page")

# Main Sections
main_sections = ["Exploratory Data Analysis", "ML Model", "NLP"]

# Subpages Mapping
subpages_mapping = {
    "Exploratory Data Analysis": [
        "EDA-Climate"
    ],
    "ML Model": [
        "model-training",
        "model-prediction"
    ],
    "NLP": [
        "Sentiment Analysis",
        "Summarize details"
        
    ],
}


# Page Function Mapping
PAGES = {
    "EDA-Climate": lambda: show_analysis(df),
    "model-training" : lambda: render(),
    "model-prediction":lambda: predict(),
    "Sentiment Analysis": lambda: st_nlp_sentiment.main() if hasattr(st_nlp_sentiment, "main") else st_nlp_sentiment(),
    "Summarize details": lambda: st_nlp_summarize.main() if hasattr(st_nlp_summarize, "main") else st_nlp_summarize()
}

# dashboard button
if st.sidebar.button("📊 My Dashboard"):
    st.session_state.main_section = "Select..."
    st.session_state.sub_page = "Select..."
    st.session_state.page = "My Dashboard"
    st.session_state.nlp_section = "Select..."

# Select Main Section
selected_main = st.sidebar.selectbox(
    "Select Section",
    ["Select..."] + main_sections,
    index=0,
    key="main_section"
)

# Select Subpage if a Main Section is selected
if selected_main != "Select...":
    available_subpages = subpages_mapping[selected_main]
    selected_subpage = st.sidebar.selectbox(
        f"Select {selected_main} Page",
        ["Select..."] + available_subpages,
        index=0,
        key="sub_page"
    )

    if selected_subpage != "Select...":
        st.session_state.page = selected_subpage



# --- Main Page Content ---
if st.session_state.page == "My Dashboard":
    st.write("""  
    ### 🌦️ Climate change Prediction Application 
 
    Navigate through the sections using the sidebar.  

    **Key Features:**
    - Exploratory data analysis
    - Model Training
    - District wise highheat, drought, glacier melting and average temperature prediction
    - Sentiment analysis and summarization using NLP
    """)
    st.markdown("---")
    st.warning("⚠️ Notice : If the page is not redirected properly, kindly refresh the browser.")
else:
    page_func = PAGES.get(st.session_state.page, None)
    if page_func:
        try:
            page_func()
        except Exception as e:
            st.error(f"❌ Error loading page `{st.session_state.page}`: {str(e)}")
    else:
        st.info(f"ℹ️ Page `{st.session_state.page}` is a placeholder. Content coming soon.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        Developed by <strong>Pratik Tamrakar</strong>  
        <br>
        🔗 <a href="https://www.linkedin.com/in/pratik-tamrakar-a2272a153/" target="_blank">LinkedIn</a> |
        💻 <a href="https://github.com/hyperactive91" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
