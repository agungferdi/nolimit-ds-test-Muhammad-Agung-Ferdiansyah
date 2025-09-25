"""
Modern Sentiment Analysis App with Hugging Face Models
A beautiful and interactive Streamlit application for sentiment analysis
with similarity search using FAISS and embeddings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time
import json
from datetime import datetime
import base64

# Add current directory and src to path for imports
sys.path.append('.')
sys.path.append('src')
from src.models.sentiment_classifier import SentimentClassifier

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular components
from utils.ui_components import load_css, render_header, render_footer
from pages.single_analysis import single_analysis_page
from pages.batch_analysis import batch_analysis_page
from pages.dataset_explorer import dataset_explorer_page
from pages.similarity_search import similarity_search_page

# Load CSS from external file
load_css()


@st.cache_resource
def load_model():
    """Load the sentiment classifier model."""
    classifier = SentimentClassifier(
        classification_model="distilbert-base-uncased-finetuned-sst-2-english",
        embedding_model="all-MiniLM-L6-v2"
    )
    return classifier


def main():
    """Main application function."""
    # Load CSS and render professional header
    load_css()
    render_header()
    
    # Load model
    with st.spinner('Loading AI models...'):
        classifier = load_model()
    
    # Sidebar navigation
    st.sidebar.markdown("## Control Panel")
    st.sidebar.markdown("---")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Single Text Analysis", "Batch Analysis", "Dataset Explorer", "Similarity Search"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    st.sidebar.info("""
    **Classification Model:**  
    distilbert-base-uncased-finetuned-sst-2-english
    
    **Embedding Model:**  
    all-MiniLM-L6-v2
    
    **Vector Search:**  
    FAISS (Facebook AI Similarity Search)
    """)
    
    # Route to appropriate page module
    if mode == "Single Text Analysis":
        single_analysis_page(classifier)
    elif mode == "Batch Analysis":
        batch_analysis_page(classifier)
    elif mode == "Dataset Explorer":
        dataset_explorer_page(classifier)
    elif mode == "Similarity Search":
        similarity_search_page(classifier)
    
    # Professional footer
    render_footer()


if __name__ == "__main__":
    main()