"""
UI Components and Utilities
Professional components for the Sentiment Analysis app
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def load_css():
    """Load the main CSS file into Streamlit."""
    css_file = Path("assets/styles.css")
    
    if css_file.exists():
        with open(css_file) as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("CSS file not found. Using default Streamlit styling.")


def get_sentiment_color(sentiment):
    """Get color for sentiment display."""
    colors = {
        'positive': '#28a745',
        'negative': '#dc3545', 
        'neutral': '#ffc107'
    }
    return colors.get(sentiment.lower(), '#6c757d')


def get_sentiment_label(sentiment):
    """Get professional text label for sentiment (no emojis)."""
    labels = {
        'positive': 'POSITIVE',
        'negative': 'NEGATIVE',
        'neutral': 'NEUTRAL'
    }
    return labels.get(sentiment.lower(), 'UNKNOWN')


def create_confidence_gauge(confidence):
    """Create a confidence gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def create_sentiment_distribution_chart(predictions):
    """Create sentiment distribution chart."""
    sentiment_counts = pd.Series(predictions).value_counts()
    colors = ['#28a745', '#dc3545', '#ffc107']
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_sequence=colors
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig


def render_sentiment_card(sentiment, confidence):
    """Render a professional sentiment analysis result card."""
    color = get_sentiment_color(sentiment)
    label = get_sentiment_label(sentiment)
    
    st.markdown(f"""
    <div class="{sentiment.lower()}-sentiment">
        <h3>Sentiment Analysis</h3>
        <h2 style="color: white;">{label}</h2>
        <p style="color: white; opacity: 0.9;">Confidence: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)


def render_similarity_card(rank, similarity_score, text):
    """Render a similarity search result card."""
    # Determine score color based on similarity
    if similarity_score > 0.7:
        score_color = "#28a745"
    elif similarity_score > 0.5:
        score_color = "#ffc107"
    else:
        score_color = "#dc3545"
    
    st.markdown(f"""
    <div class="similarity-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong>Rank #{rank}</strong>
            <span style="color: {score_color}; font-weight: bold;">
                {similarity_score:.4f} similarity
            </span>
        </div>
        <p style="margin-top: 10px; margin-bottom: 0;">{text}</p>
    </div>
    """, unsafe_allow_html=True)


def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-header">AI Sentiment Analysis Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Hugging Face Transformers & FAISS Similarity Search</p>', unsafe_allow_html=True)


def render_footer():
    """Render the application footer."""
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>AI Sentiment Analysis Studio</strong></p>
        <p>Built with Hugging Face Transformers, FAISS, and Streamlit</p>
        <p><em>NoLimit Indonesia - Data Scientist Hiring Test</em></p>
        <p>Created by: Muhammad Agung Ferdiansyah | 2024</p>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample dataset."""
    try:
        df = pd.read_csv('data/sample_reviews.csv')
        return df
    except FileNotFoundError:
        st.error("Sample data not found. Please ensure 'data/sample_reviews.csv' exists.")
        return pd.DataFrame()