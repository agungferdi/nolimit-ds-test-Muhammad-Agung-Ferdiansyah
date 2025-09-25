"""
Single Text Analysis Page
Professional sentiment analysis for individual texts
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_components import (
    get_sentiment_label, render_sentiment_card, 
    create_confidence_gauge
)


def single_analysis_page(classifier):
    """Single text analysis interface."""
    st.markdown("## Single Text Analysis")
    
    # Text input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter your text for sentiment analysis:",
            placeholder="Type or paste your text here... (e.g., movie review, tweet, comment)",
            height=150
        )
    
    with col2:
        st.markdown("### Quick Examples")
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
    
    if st.button("Analyze Sentiment", type="primary"):
        if text_input.strip():
            with st.spinner('AI is analyzing...'):
                result = classifier.predict_sentiment(text_input)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            sentiment = result['sentiment']
            confidence = result['confidence']
            
            with col1:
                render_sentiment_card(sentiment, confidence)
            
            with col2:
                st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
            
            with col3:
                st.markdown("### All Scores")
                for sent, score in result['all_scores'].items():
                    st.metric(
                        sent.capitalize(),
                        f"{score:.1%}",
                        delta=f"{score - (1/3):.1%}"
                    )
            
            # Detailed breakdown
            st.markdown("---")
            st.markdown("### Detailed Analysis")
            
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
            st.warning("Please enter some text to analyze.")