"""
Dataset Explorer Page
Professional dataset exploration interface
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_components import get_sentiment_label, render_sentiment_card
from utils.data_utils import load_sample_data


def dataset_explorer_page(classifier):
    """Dataset exploration interface."""
    st.markdown("## Dataset Explorer")
    
    sample_df = load_sample_data()
    
    if sample_df.empty:
        st.error("No dataset available for exploration.")
        return
    
    # Dataset overview
    st.markdown("### Dataset Overview")
    
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
    st.markdown("### Sample Data")
    
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
        label_text = get_sentiment_label(row['label'])
        with st.expander(f"{label_text} {row['text'][:80]}..."):
            st.write(f"**Full Text:** {row['text']}")
            st.write(f"**Label:** {row['label']}")
            st.write(f"**Length:** {len(row['text'])} characters")
            
            # Quick analysis
            if st.button(f"Quick Analysis", key=f"analyze_{idx}"):
                result = classifier.predict_sentiment(row['text'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Predicted:** {result['sentiment']}")
                    st.write(f"**Confidence:** {result['confidence']:.4f}")
                
                with col2:
                    match = "Match" if result['sentiment'] == row['label'] else "Mismatch"
                    st.write(f"**Accuracy:** {match}")