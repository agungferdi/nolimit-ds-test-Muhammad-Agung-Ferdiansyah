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

# Add src to path
sys.path.append('src')
from models.sentiment_classifier import SentimentClassifier

# Page configuration
st.set_page_config(
    page_title="🎭 Sentiment Analysis AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .positive-sentiment {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    
    .negative-sentiment {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    
    .neutral-sentiment {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .similarity-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the sentiment classifier model."""
    classifier = SentimentClassifier(
        classification_model="distilbert-base-uncased-finetuned-sst-2-english",
        embedding_model="all-MiniLM-L6-v2"
    )
    return classifier


@st.cache_data
def load_sample_data():
    """Load sample dataset."""
    try:
        df = pd.read_csv('data/sample_reviews.csv')
        return df
    except FileNotFoundError:
        st.error("Sample data not found. Please ensure 'data/sample_reviews.csv' exists.")
        return pd.DataFrame()


def get_sentiment_color(sentiment):
    """Get color for sentiment display."""
    colors = {
        'positive': '#28a745',
        'negative': '#dc3545', 
        'neutral': '#ffc107'
    }
    return colors.get(sentiment.lower(), '#6c757d')


def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment."""
    emojis = {
        'positive': '😊',
        'negative': '😞',
        'neutral': '😐'
    }
    return emojis.get(sentiment.lower(), '❓')


def create_confidence_gauge(confidence):
    """Create a confidence gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence"},
        delta = {'reference': 50},
        gauge = {
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


def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 AI Sentiment Analysis Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Hugging Face Transformers & FAISS Similarity Search</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('🤖 Loading AI models...'):
        classifier = load_model()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    st.sidebar.markdown("---")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "🔧 Choose Mode",
        ["Single Text Analysis", "Batch Analysis", "Dataset Explorer", "Similarity Search"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.info(f"""
    **Classification Model:**  
    `cardiffnlp/twitter-roberta-base-sentiment-latest`
    
    **Embedding Model:**  
    `all-MiniLM-L6-v2`
    
    **Vector Search:**  
    FAISS (Facebook AI Similarity Search)
    """)
    
    # Main content based on mode
    if mode == "Single Text Analysis":
        single_text_analysis(classifier)
    elif mode == "Batch Analysis":
        batch_analysis(classifier)
    elif mode == "Dataset Explorer":
        dataset_explorer(classifier)
    elif mode == "Similarity Search":
        similarity_search_mode(classifier)


def single_text_analysis(classifier):
    """Single text analysis interface."""
    st.markdown("## 🔍 Single Text Analysis")
    
    # Text input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "📝 Enter your text for sentiment analysis:",
            placeholder="Type or paste your text here... (e.g., movie review, tweet, comment)",
            height=150
        )
    
    with col2:
        st.markdown("### 🎯 Quick Examples")
        examples = [
            "This movie is absolutely fantastic!",
            "Terrible film, complete waste of time.",
            "The movie was okay, nothing special.",
            "I'm not sure what to think about this."
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                text_input = example
                st.rerun()
    
    if st.button("🚀 Analyze Sentiment", type="primary"):
        if text_input.strip():
            with st.spinner('🧠 AI is analyzing...'):
                result = classifier.predict_sentiment(text_input)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            sentiment = result['sentiment']
            confidence = result['confidence']
            emoji = get_sentiment_emoji(sentiment)
            color = get_sentiment_color(sentiment)
            
            with col1:
                st.markdown(f"""
                <div class="{sentiment.lower()}-sentiment">
                    <h3>{emoji} Sentiment</h3>
                    <h2 style="color: white;">{sentiment.upper()}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
            
            with col3:
                # All scores
                st.markdown("### 📊 All Scores")
                for sent, score in result['all_scores'].items():
                    st.metric(
                        sent.capitalize(),
                        f"{score:.1%}",
                        delta=f"{score - (1/3):.1%}"
                    )
            
            # Detailed breakdown
            st.markdown("---")
            st.markdown("### 📋 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "sentiment": sentiment,
                    "confidence": f"{confidence:.4f}",
                    "interpretation": f"The AI is {confidence:.1%} confident that this text expresses {sentiment} sentiment."
                })
            
            with col2:
                # Score visualization
                scores_df = pd.DataFrame(list(result['all_scores'].items()), columns=['Sentiment', 'Score'])
                fig = px.bar(scores_df, x='Sentiment', y='Score', 
                           title="Sentiment Scores Breakdown",
                           color='Score',
                           color_continuous_scale='RdYlGn')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Please enter some text to analyze.")


def batch_analysis(classifier):
    """Batch analysis interface."""
    st.markdown("## 📊 Batch Text Analysis")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Upload CSV File", "Paste Multiple Texts", "Use Sample Data"]
    )
    
    texts_to_analyze = []
    
    if input_method == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if 'text' in df.columns:
                texts_to_analyze = df['text'].tolist()
            else:
                text_column = st.selectbox("Select text column:", df.columns)
                texts_to_analyze = df[text_column].tolist()
    
    elif input_method == "Paste Multiple Texts":
        text_area = st.text_area(
            "Enter multiple texts (one per line):",
            placeholder="Text 1\nText 2\nText 3...",
            height=200
        )
        if text_area:
            texts_to_analyze = [text.strip() for text in text_area.split('\n') if text.strip()]
    
    elif input_method == "Use Sample Data":
        sample_df = load_sample_data()
        if not sample_df.empty:
            st.write("Using sample movie reviews data:")
            st.dataframe(sample_df.head(10))
            texts_to_analyze = sample_df['text'].tolist()[:20]  # Limit for demo
    
    if texts_to_analyze and st.button("🚀 Analyze All Texts", type="primary"):
        with st.spinner(f'🧠 Analyzing {len(texts_to_analyze)} texts...'):
            results = classifier.predict_batch(texts_to_analyze)
        
        # Process results
        results_df = pd.DataFrame(results)
        
        # Overview metrics
        st.markdown("### 📈 Analysis Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Texts", len(results_df))
        
        with col2:
            avg_confidence = results_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            most_common = results_df['sentiment'].mode()[0]
            st.metric("Most Common", most_common.title())
        
        with col4:
            high_confidence = (results_df['confidence'] > 0.8).sum()
            st.metric("High Confidence", f"{high_confidence}/{len(results_df)}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            fig = create_sentiment_distribution_chart(results_df['sentiment'].tolist())
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig = px.histogram(
                results_df, x='confidence', nbins=20,
                title="Confidence Score Distribution",
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.markdown("### 📋 Detailed Results")
        
        # Add filtering
        sentiment_filter = st.selectbox(
            "Filter by sentiment:",
            ["All"] + list(results_df['sentiment'].unique())
        )
        
        if sentiment_filter != "All":
            filtered_df = results_df[results_df['sentiment'] == sentiment_filter]
        else:
            filtered_df = results_df
        
        # Display with styling
        for idx, row in filtered_df.iterrows():
            with st.expander(f"{get_sentiment_emoji(row['sentiment'])} {row['text'][:100]}... (Confidence: {row['confidence']:.1%})"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Full Text:** {row['text']}")
                    st.write(f"**Sentiment:** {row['sentiment'].title()}")
                    st.write(f"**Confidence:** {row['confidence']:.4f}")
                
                with col2:
                    # Individual scores
                    if 'all_scores' in row and row['all_scores']:
                        st.write("**All Scores:**")
                        for sent, score in row['all_scores'].items():
                            st.write(f"- {sent}: {score:.3f}")


def dataset_explorer(classifier):
    """Dataset exploration interface."""
    st.markdown("## 🗂️ Dataset Explorer")
    
    sample_df = load_sample_data()
    
    if sample_df.empty:
        st.error("No dataset available for exploration.")
        return
    
    # Dataset overview
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(sample_df))
    with col2:
        st.metric("Unique Labels", sample_df['label'].nunique())
    with col3:
        avg_length = sample_df['text'].str.len().mean()
        st.metric("Avg Text Length", f"{avg_length:.0f}")
    with col4:
        st.metric("Columns", len(sample_df.columns))
    
    # Dataset statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Label distribution
        label_counts = sample_df['label'].value_counts()
        fig = px.bar(
            x=label_counts.index, y=label_counts.values,
            title="Label Distribution in Dataset",
            color=label_counts.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Text length distribution
        sample_df['text_length'] = sample_df['text'].str.len()
        fig = px.histogram(
            sample_df, x='text_length', color='label',
            title="Text Length Distribution by Label",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data viewer
    st.markdown("### 📋 Sample Data")
    
    # Filtering options
    col1, col2 = st.columns(2)
    with col1:
        selected_label = st.selectbox("Filter by label:", ["All"] + list(sample_df['label'].unique()))
    with col2:
        num_samples = st.slider("Number of samples to show:", 5, 50, 10)
    
    # Apply filters
    if selected_label != "All":
        filtered_df = sample_df[sample_df['label'] == selected_label]
    else:
        filtered_df = sample_df
    
    # Display samples
    display_df = filtered_df.head(num_samples)
    
    for idx, row in display_df.iterrows():
        with st.expander(f"{get_sentiment_emoji(row['label'])} {row['text'][:80]}..."):
            st.write(f"**Full Text:** {row['text']}")
            st.write(f"**Label:** {row['label']}")
            st.write(f"**Length:** {len(row['text'])} characters")
            
            # Quick analysis
            if st.button(f"🔍 Quick Analysis", key=f"analyze_{idx}"):
                result = classifier.predict_sentiment(row['text'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Predicted:** {result['sentiment']}")
                    st.write(f"**Confidence:** {result['confidence']:.4f}")
                
                with col2:
                    match = "✅ Match" if result['sentiment'] == row['label'] else "❌ Mismatch"
                    st.write(f"**Accuracy:** {match}")


def similarity_search_mode(classifier):
    """Similarity search interface."""
    st.markdown("## 🔍 Similarity Search")
    st.markdown("Find texts similar to your input using AI embeddings and FAISS search.")
    
    # Load and prepare data for similarity search
    sample_df = load_sample_data()
    
    if sample_df.empty:
        st.error("No dataset available for similarity search.")
        return
    
    # Build FAISS index if not already built
    if classifier.faiss_index is None:
        with st.spinner('🔧 Building FAISS search index...'):
            texts = sample_df['text'].tolist()
            classifier.build_faiss_index(texts)
        st.success("✅ Search index built successfully!")
    
    # Query input
    query_text = st.text_area(
        "🔍 Enter your query text:",
        placeholder="Enter text to find similar examples...",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        num_results = st.slider("Number of similar texts to find:", 1, 20, 5)
    with col2:
        include_sentiment = st.checkbox("Include sentiment analysis", value=True)
    
    if st.button("🚀 Find Similar Texts", type="primary") and query_text.strip():
        with st.spinner('🔍 Searching for similar texts...'):
            if include_sentiment:
                results = classifier.analyze_sentiment_with_similarity(query_text, k=num_results)
                sentiment_analysis = results['sentiment_analysis']
                similar_texts = results['similar_texts']
            else:
                similar_texts = classifier.find_similar_texts(query_text, k=num_results)
                sentiment_analysis = None
        
        # Display query analysis
        if sentiment_analysis:
            st.markdown("### 🎯 Query Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment = sentiment_analysis['sentiment']
                emoji = get_sentiment_emoji(sentiment)
                st.markdown(f"""
                <div class="{sentiment.lower()}-sentiment" style="text-align: center;">
                    <h4>{emoji} {sentiment.upper()}</h4>
                    <p>Confidence: {sentiment_analysis['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fig = create_confidence_gauge(sentiment_analysis['confidence'])
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                scores_data = sentiment_analysis['all_scores']
                for sent, score in scores_data.items():
                    st.metric(sent.capitalize(), f"{score:.1%}")
        
        # Display similar texts
        st.markdown("### 🎯 Similar Texts Found")
        
        if similar_texts:
            # Create similarity score chart
            similarity_df = pd.DataFrame(similar_texts)
            
            fig = px.bar(
                similarity_df, x='rank', y='similarity_score',
                title="Similarity Scores",
                hover_data=['text'],
                color='similarity_score',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display each similar text
            for result in similar_texts:
                similarity_score = result['similarity_score']
                
                # Color coding based on similarity
                if similarity_score > 0.7:
                    card_class = "similarity-card" 
                    score_color = "#28a745"
                elif similarity_score > 0.5:
                    score_color = "#ffc107"
                else:
                    score_color = "#dc3545"
                
                st.markdown(f"""
                <div class="similarity-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>Rank #{result['rank']}</strong>
                        <span style="color: {score_color}; font-weight: bold;">
                            {similarity_score:.4f} similarity
                        </span>
                    </div>
                    <p style="margin-top: 10px; margin-bottom: 0;">{result['text']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Option to analyze each similar text
                with st.expander(f"🔍 Analyze Similar Text #{result['rank']}"):
                    similar_result = classifier.predict_sentiment(result['text'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Sentiment:** {similar_result['sentiment']}")
                        st.write(f"**Confidence:** {similar_result['confidence']:.4f}")
                    
                    with col2:
                        st.write("**All Scores:**")
                        for sent, score in similar_result['all_scores'].items():
                            st.write(f"- {sent}: {score:.3f}")
        
        else:
            st.warning("No similar texts found. Try a different query.")


# Footer
def add_footer():
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🎭 <strong>AI Sentiment Analysis Studio</strong></p>
        <p>Built with ❤️ using Hugging Face Transformers, FAISS, and Streamlit</p>
        <p><em>NoLimit Indonesia - Data Scientist Hiring Test</em></p>
        <p>Created by: Ferdiansyah Muhammad Agung | 2024</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    add_footer()