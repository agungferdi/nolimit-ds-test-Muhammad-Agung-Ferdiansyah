"""
Batch Analysis Page
Professional batch sentiment analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_components import (
    get_sentiment_label, render_sentiment_card,
    create_sentiment_distribution_chart
)
from utils.data_utils import load_sample_data


def batch_analysis_page(classifier):
    """Batch analysis interface."""
    st.markdown("## Batch Text Analysis")
    
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
    
    if texts_to_analyze and st.button("Analyze All Texts", type="primary"):
        with st.spinner(f'Analyzing {len(texts_to_analyze)} texts...'):
            results = classifier.predict_batch(texts_to_analyze)
        
        # Process results
        results_df = pd.DataFrame(results)
        
        # Overview metrics
        st.markdown("### Analysis Overview")
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
            fig = create_sentiment_distribution_chart(results_df['sentiment'].tolist())
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                results_df, x='confidence', nbins=20,
                title="Confidence Score Distribution",
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Results display
        st.markdown("### Detailed Results")
        
        sentiment_filter = st.selectbox(
            "Filter by sentiment:",
            ["All"] + list(results_df['sentiment'].unique())
        )
        
        if sentiment_filter != "All":
            filtered_df = results_df[results_df['sentiment'] == sentiment_filter]
        else:
            filtered_df = results_df
        
        # Display results
        for idx, row in filtered_df.iterrows():
            sentiment_label = get_sentiment_label(row['sentiment'])
            with st.expander(f"{sentiment_label} {row['text'][:100]}... (Confidence: {row['confidence']:.1%})"):
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