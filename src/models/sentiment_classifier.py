"""
Sentiment Analysis Classification using Hugging Face Models
This module implements sentiment analysis with embeddings and similarity search.
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentClassifier:
    """
    Advanced Sentiment Analysis Classifier using Hugging Face models
    with embedding-based similarity search capabilities.
    """
    
    def __init__(self, 
                 classification_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentiment classifier with specified models.
        
        Args:
            classification_model: HuggingFace model for classification
            embedding_model: SentenceTransformer model for embeddings
        """
        self.classification_model_name = classification_model
        self.embedding_model_name = embedding_model
        
        # Initialize models
        logger.info("Loading classification model...")
        self.tokenizer = AutoTokenizer.from_pretrained(classification_model)
        self.classifier = pipeline(
            "sentiment-analysis",
            model=classification_model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )
        
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize FAISS index
        self.faiss_index = None
        self.texts_database = []
        self.embeddings_database = None
        
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text for sentiment analysis
            
        Returns:
            Dictionary containing sentiment prediction and confidence scores
        """
        try:
            # Get predictions from the model
            results = self.classifier(text)
            
            # Process results
            predictions = results[0] if isinstance(results, list) else results
            
            # Find the highest confidence prediction
            best_prediction = max(predictions, key=lambda x: x['score'])
            
            # Normalize label names
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive'
            }
            
            normalized_label = label_mapping.get(
                best_prediction['label'].upper(), 
                best_prediction['label'].lower()
            )
            
            return {
                'text': text,
                'sentiment': normalized_label,
                'confidence': best_prediction['score'],
                'all_scores': {
                    label_mapping.get(pred['label'].upper(), pred['label'].lower()): pred['score'] 
                    for pred in predictions
                }
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {str(e)}")
            return {
                'text': text,
                'sentiment': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            results.append(self.predict_sentiment(text))
        return results
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_faiss_index(self, texts: List[str], embeddings: np.ndarray = None):
        """
        Build FAISS index for similarity search.
        
        Args:
            texts: List of texts to index
            embeddings: Pre-computed embeddings (optional)
        """
        if embeddings is None:
            embeddings = self.create_embeddings(texts)
        
        # Store texts and embeddings
        self.texts_database = texts
        self.embeddings_database = embeddings
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {len(texts)} texts")
    
    def find_similar_texts(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar texts using FAISS similarity search.
        
        Args:
            query_text: Query text to find similarities for
            k: Number of similar texts to return
            
        Returns:
            List of similar texts with scores
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index first.")
        
        # Create embedding for query
        query_embedding = self.embedding_model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Search similar texts
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.texts_database):
                results.append({
                    'rank': i + 1,
                    'text': self.texts_database[idx],
                    'similarity_score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def analyze_sentiment_with_similarity(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """
        Analyze sentiment and find similar texts in one go.
        
        Args:
            query_text: Input text to analyze
            k: Number of similar texts to find
            
        Returns:
            Combined analysis results
        """
        # Get sentiment prediction
        sentiment_result = self.predict_sentiment(query_text)
        
        # Find similar texts if FAISS index is available
        similar_texts = []
        if self.faiss_index is not None:
            try:
                similar_texts = self.find_similar_texts(query_text, k)
            except Exception as e:
                logger.warning(f"Could not find similar texts: {str(e)}")
        
        return {
            'sentiment_analysis': sentiment_result,
            'similar_texts': similar_texts,
            'query_text': query_text
        }
    
    def save_model(self, path: str):
        """Save the FAISS index and text database."""
        os.makedirs(path, exist_ok=True)
        
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, os.path.join(path, 'faiss_index.bin'))
        
        with open(os.path.join(path, 'texts_database.pkl'), 'wb') as f:
            pickle.dump(self.texts_database, f)
        
        with open(os.path.join(path, 'embeddings_database.pkl'), 'wb') as f:
            pickle.dump(self.embeddings_database, f)
            
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the FAISS index and text database."""
        if os.path.exists(os.path.join(path, 'faiss_index.bin')):
            self.faiss_index = faiss.read_index(os.path.join(path, 'faiss_index.bin'))
        
        if os.path.exists(os.path.join(path, 'texts_database.pkl')):
            with open(os.path.join(path, 'texts_database.pkl'), 'rb') as f:
                self.texts_database = pickle.load(f)
        
        if os.path.exists(os.path.join(path, 'embeddings_database.pkl')):
            with open(os.path.join(path, 'embeddings_database.pkl'), 'rb') as f:
                self.embeddings_database = pickle.load(f)
                
        logger.info(f"Model loaded from {path}")


def evaluate_model(classifier: SentimentClassifier, test_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate the sentiment classifier on test data.
    
    Args:
        classifier: Trained sentiment classifier
        test_data: DataFrame with 'text' and 'label' columns
        
    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Get predictions
    predictions = []
    true_labels = []
    
    for _, row in test_data.iterrows():
        pred = classifier.predict_sentiment(row['text'])
        predictions.append(pred['sentiment'])
        true_labels.append(row['label'])
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': predictions,
        'true_labels': true_labels
    }


if __name__ == "__main__":
    # Example usage
    classifier = SentimentClassifier()
    
    # Test single prediction
    test_text = "I love this movie! It's absolutely fantastic and well-made."
    result = classifier.predict_sentiment(test_text)
    print(f"Sentiment Analysis Result: {result}")
    
    # Test batch prediction
    test_texts = [
        "This movie is terrible, I hate it!",
        "The film was okay, nothing special.",
        "Amazing cinematography and great acting!"
    ]
    
    batch_results = classifier.predict_batch(test_texts)
    for i, result in enumerate(batch_results):
        print(f"Text {i+1}: {result['sentiment']} (confidence: {result['confidence']:.3f})")