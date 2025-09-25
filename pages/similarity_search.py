"""
Similarity Search Page
Professional similarity search with FAISS
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_components import (
    get_sentiment_label, render_sentiment_card,
    create_confidence_gauge, render_similarity_card
)
from utils.data_utils import load_sample_data


def similarity_search_page(classifier):
    """Similarity search interface."""
    st.markdown("## Similarity Search")
    st.markdown("Find texts similar to your input using AI embeddings and FAISS search.")
    
    # Load and prepare data for similarity search
    sample_df = load_sample_data()
    
    if sample_df.empty:
        st.error("No dataset available for similarity search.")
        return
    
    # Build FAISS index if not already built
    if classifier.faiss_index is None:
        with st.spinner('Building FAISS search index...'):
            texts = sample_df['text'].tolist()
            classifier.build_faiss_index(texts)
        st.success("Search index built successfully!")
    
    # Query input
    query_text = st.text_area(
        "Enter your query text:",
        placeholder="Enter text to find similar examples...",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        num_results = st.slider("Number of similar texts to find:", 1, 20, 5)
    with col2:
        include_sentiment = st.checkbox("Include sentiment analysis", value=True)
    
    if st.button("Find Similar Texts", type="primary") and query_text.strip():
        with st.spinner('Searching for similar texts...'):
            if include_sentiment:
                results = classifier.analyze_sentiment_with_similarity(query_text, k=num_results)
                sentiment_analysis = results['sentiment_analysis']
                similar_texts = results['similar_texts']
            else:
                similar_texts = classifier.find_similar_texts(query_text, k=num_results)
                sentiment_analysis = None
        
        # Display query analysis
        if sentiment_analysis:
            st.markdown("### Query Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment = sentiment_analysis['sentiment']
                confidence = sentiment_analysis['confidence']
                render_sentiment_card(sentiment, confidence)
            
            with col2:
                fig = create_confidence_gauge(sentiment_analysis['confidence'])
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                scores_data = sentiment_analysis['all_scores']
                for sent, score in scores_data.items():
                    st.metric(sent.capitalize(), f"{score:.1%}")
        
        # Display similar texts
        st.markdown("### Similar Texts Found")
        
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
            
            # Display each similar text using professional components
            for result in similar_texts:
                render_similarity_card(result['rank'], result['similarity_score'], result['text'])
                
                # Option to analyze each similar text
                with st.expander(f"Analyze Similar Text #{result['rank']}"):
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