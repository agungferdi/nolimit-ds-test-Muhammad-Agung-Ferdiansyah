"""
Utility functions for data processing and management
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
import requests
import os
from datasets import load_dataset
import random

logger = logging.getLogger(__name__)

def create_sample_dataset(output_path: str, num_samples: int = 1000) -> pd.DataFrame:
    """
    Create a sample dataset of movie reviews for testing.
    
    Args:
        output_path: Path to save the dataset
        num_samples: Number of samples to include
        
    Returns:
        DataFrame with sample data
    """
    logger.info(f"Creating sample dataset with {num_samples} samples...")
    
    # Sample movie reviews data
    positive_reviews = [
        "This movie is absolutely fantastic! Great acting and storyline.",
        "I loved every minute of it. Brilliant cinematography and direction.",
        "One of the best films I've ever seen. Highly recommended!",
        "Amazing performances by all actors. A masterpiece!",
        "Incredible visual effects and compelling narrative.",
        "Outstanding movie with excellent character development.",
        "A truly inspiring and beautiful film experience.",
        "Superb acting and an engaging plot throughout.",
        "This film exceeded all my expectations. Perfect!",
        "Wonderfully crafted with great attention to detail.",
        "The best movie of the year! Absolutely stunning.",
        "Brilliant script and phenomenal performances.",
        "A cinematic gem that will stay with you forever.",
        "Expertly directed with incredible emotional depth.",
        "This movie is pure gold. Cannot recommend enough!",
        "Fascinating story with incredible visual storytelling.",
        "The acting is top-notch and the story is captivating.",
        "A beautiful and moving film that touches your heart.",
        "Excellent production values and stellar performances.",
        "This film is a work of art. Simply magnificent!"
    ]
    
    negative_reviews = [
        "This movie is terrible. Waste of time and money.",
        "Boring plot and poor acting. Very disappointing.",
        "I couldn't even finish watching it. Awful!",
        "Completely overrated and poorly executed.",
        "The worst film I've seen this year.",
        "Terrible script and unconvincing performances.",
        "A complete disaster from start to finish.",
        "Poorly directed with no coherent storyline.",
        "Absolutely horrible. Don't bother watching.",
        "This movie is a complete failure in every aspect.",
        "Incredibly boring and predictable plot.",
        "Poor character development and weak dialogue.",
        "A disappointing mess with no redeeming qualities.",
        "Terrible acting and confusing narrative.",
        "This film is painful to watch. Avoid at all costs!",
        "Poorly written with unconvincing character arcs.",
        "A boring and uninspired piece of cinema.",
        "Completely lacks substance and entertainment value.",
        "Awful direction and terrible pacing throughout.",
        "This movie is a waste of everyone's time."
    ]
    
    neutral_reviews = [
        "The movie was okay, nothing particularly special.",
        "It's an average film with some good and bad parts.",
        "Not bad, but not great either. Just mediocre.",
        "The movie is fine, but I've seen better.",
        "It's watchable but forgettable entertainment.",
        "The film has its moments but overall is average.",
        "Decent movie but nothing to write home about.",
        "It's an okay film with some interesting elements.",
        "The movie is alright, though it could be better.",
        "Moderately entertaining but not exceptional.",
        "A reasonable film with mixed results.",
        "The movie is adequate but lacks wow factor.",
        "It's a standard film with typical storytelling.",
        "The movie is fair but doesn't stand out.",
        "Acceptable entertainment with room for improvement.",
        "The film is passable but not memorable.",
        "It's an average production with standard quality.",
        "The movie is tolerable but uninspiring.",
        "Decent enough but nothing groundbreaking.",
        "The film is satisfactory but not remarkable."
    ]
    
    # Create balanced dataset
    samples_per_class = num_samples // 3
    
    data = []
    
    # Add positive samples
    for i in range(samples_per_class):
        review = positive_reviews[i % len(positive_reviews)]
        # Add some variation
        if i >= len(positive_reviews):
            review = f"Variation {i//len(positive_reviews)}: " + review
        data.append({"text": review, "label": "positive"})
    
    # Add negative samples
    for i in range(samples_per_class):
        review = negative_reviews[i % len(negative_reviews)]
        if i >= len(negative_reviews):
            review = f"Variation {i//len(negative_reviews)}: " + review
        data.append({"text": review, "label": "negative"})
    
    # Add neutral samples
    remaining_samples = num_samples - 2 * samples_per_class
    for i in range(remaining_samples):
        review = neutral_reviews[i % len(neutral_reviews)]
        if i >= len(neutral_reviews):
            review = f"Variation {i//len(neutral_reviews)}: " + review
        data.append({"text": review, "label": "neutral"})
    
    # Shuffle the data
    random.shuffle(data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample dataset created and saved to {output_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def load_imdb_dataset(num_samples: int = 2000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load IMDB movie reviews dataset from Hugging Face.
    
    Args:
        num_samples: Number of samples per split
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("Loading IMDB dataset from Hugging Face...")
    
    try:
        # Load dataset
        dataset = load_dataset("imdb")
        
        # Convert to DataFrames
        train_data = dataset['train'].to_pandas()
        test_data = dataset['test'].to_pandas()
        
        # Map labels
        label_mapping = {0: 'negative', 1: 'positive'}
        train_data['label'] = train_data['label'].map(label_mapping)
        test_data['label'] = test_data['label'].map(label_mapping)
        
        # Sample data if requested
        if num_samples < len(train_data):
            train_data = train_data.sample(n=num_samples, random_state=42)
        if num_samples < len(test_data):
            test_data = test_data.sample(n=num_samples, random_state=42)
        
        logger.info(f"Loaded IMDB dataset - Train: {len(train_data)}, Test: {len(test_data)}")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Failed to load IMDB dataset: {str(e)}")
        logger.info("Using sample dataset instead...")
        
        # Fallback to sample dataset
        sample_df = create_sample_dataset("data/sample_reviews.csv", num_samples)
        
        # Split into train/test
        train_size = int(0.8 * len(sample_df))
        train_df = sample_df[:train_size].reset_index(drop=True)
        test_df = sample_df[train_size:].reset_index(drop=True)
        
        return train_df, test_df


def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string and basic cleaning
    text = str(text).strip()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio for training data
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    train_size = int(len(df_shuffled) * train_ratio)
    train_df = df_shuffled[:train_size]
    test_df = df_shuffled[train_size:]
    
    return train_df, test_df


def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get statistics about the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_samples': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Add label distribution if label column exists
    if 'label' in df.columns:
        stats['label_distribution'] = df['label'].value_counts().to_dict()
    
    # Add text length statistics if text column exists
    if 'text' in df.columns:
        text_lengths = df['text'].str.len()
        stats['text_length_stats'] = {
            'mean': float(text_lengths.mean()),
            'median': float(text_lengths.median()),
            'min': int(text_lengths.min()),
            'max': int(text_lengths.max()),
            'std': float(text_lengths.std())
        }
    
    return stats


def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for the application.
    
    Returns:
        DataFrame with sample movie reviews
    """
    # Check if sample data file exists
    sample_file = "data/sample_reviews.csv"
    
    if os.path.exists(sample_file):
        try:
            df = pd.read_csv(sample_file)
            logger.info(f"Loaded sample data from {sample_file}")
            return df
        except Exception as e:
            logger.warning(f"Could not load {sample_file}: {str(e)}")
    
    # Create sample data if file doesn't exist
    logger.info("Creating new sample dataset...")
    df = create_sample_dataset(sample_file, 100)
    return df


if __name__ == "__main__":
    # Example usage
    
    # Create sample dataset
    sample_df = create_sample_dataset("../data/sample_reviews.csv", 300)
    print("Sample dataset created!")
    
    # Get dataset statistics
    stats = get_dataset_stats(sample_df)
    print(f"Dataset statistics: {stats}")
    
    # Try loading IMDB dataset
    try:
        train_df, test_df = load_imdb_dataset(100)
        print(f"IMDB dataset loaded - Train: {len(train_df)}, Test: {len(test_df)}")
    except Exception as e:
        print(f"Could not load IMDB dataset: {e}")