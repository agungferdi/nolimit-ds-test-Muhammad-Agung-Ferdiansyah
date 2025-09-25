"""
Main Script - Sentiment Analysis Classification
NoLimit Indonesia Data Scientist Hiring Test

This script demonstrates the complete sentiment analysis pipeline
using Hugging Face models and FAISS similarity search.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from models.sentiment_classifier import SentimentClassifier
from utils.data_utils import create_sample_dataset, get_dataset_stats
import json
import time
from datetime import datetime

def main():
    """Main function demonstrating the sentiment analysis pipeline."""
    print("🎭 Sentiment Analysis Classification Pipeline")
    print("=" * 60)
    print("NoLimit Indonesia - Data Scientist Hiring Test")
    print("Task A: Classification using Hugging Face Models")
    print("=" * 60)
    
    # 1. Initialize the sentiment classifier
    print("\n1. 🤖 Initializing Sentiment Classifier...")
    print("   Loading Hugging Face models...")
    
    start_time = time.time()
    classifier = SentimentClassifier(
        classification_model="distilbert-base-uncased-finetuned-sst-2-english",
        embedding_model="all-MiniLM-L6-v2"
    )
    load_time = time.time() - start_time
    
    print(f"   ✅ Models loaded successfully! ({load_time:.2f}s)")
    print(f"   📊 Classification Model: {classifier.classification_model_name}")
    print(f"   🔮 Embedding Model: {classifier.embedding_model_name}")
    
    # 2. Load or create sample dataset
    print("\n2. 📊 Loading Dataset...")
    
    data_path = 'data/sample_reviews.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"   ✅ Loaded existing dataset: {data_path}")
    else:
        print("   📝 Creating sample dataset...")
        df = create_sample_dataset(data_path, num_samples=100)
        print(f"   ✅ Sample dataset created: {data_path}")
    
    # Dataset statistics
    stats = get_dataset_stats(df)
    print(f"   📈 Dataset Statistics:")
    print(f"      • Total samples: {stats['total_samples']}")
    print(f"      • Label distribution: {stats.get('label_distribution', 'N/A')}")
    if 'text_length_stats' in stats:
        print(f"      • Avg text length: {stats['text_length_stats']['mean']:.1f} chars")
    
    # 3. Demonstrate single text prediction
    print("\n3. 🔍 Single Text Prediction Examples...")
    
    example_texts = [
        "This movie is absolutely fantastic! Best film I've ever seen.",
        "Terrible movie, complete waste of time and money.",
        "The film was okay, nothing particularly special.",
        "Mixed feelings about this one - some parts were good, others not so much."
    ]
    
    predictions = []
    for i, text in enumerate(example_texts, 1):
        print(f"\n   Example {i}: {text}")
        
        result = classifier.predict_sentiment(text)
        predictions.append(result)
        
        print(f"   → Sentiment: {result['sentiment'].upper()}")
        print(f"   → Confidence: {result['confidence']:.4f}")
        print(f"   → All Scores: {result['all_scores']}")
    
    # 4. Create embeddings and build FAISS index
    print("\n4. 🔮 Creating Embeddings and Building FAISS Index...")
    
    all_texts = df['text'].tolist()
    
    print(f"   Creating embeddings for {len(all_texts)} texts...")
    start_time = time.time()
    embeddings = classifier.create_embeddings(all_texts)
    embedding_time = time.time() - start_time
    
    print(f"   ✅ Embeddings created! ({embedding_time:.2f}s)")
    print(f"   📐 Embedding dimensions: {embeddings.shape[1]}")
    
    print("   Building FAISS index for similarity search...")
    classifier.build_faiss_index(all_texts, embeddings)
    print("   ✅ FAISS index built successfully!")
    
    # 5. Demonstrate similarity search
    print("\n5. 🔍 Similarity Search Demonstration...")
    
    query_text = "This movie is amazing and wonderful!"
    print(f"   Query: '{query_text}'")
    
    # Combined analysis
    combined_result = classifier.analyze_sentiment_with_similarity(query_text, k=5)
    
    # Show sentiment analysis
    sentiment_result = combined_result['sentiment_analysis']
    print(f"   🎯 Sentiment Analysis:")
    print(f"      • Predicted sentiment: {sentiment_result['sentiment']}")
    print(f"      • Confidence: {sentiment_result['confidence']:.4f}")
    
    # Show similar texts
    similar_texts = combined_result['similar_texts']
    print(f"   🔍 Top {len(similar_texts)} Similar Texts:")
    for result in similar_texts:
        print(f"      {result['rank']}. Score: {result['similarity_score']:.4f}")
        print(f"         Text: {result['text']}")
    
    # 6. Batch prediction and evaluation
    print("\n6. 📊 Batch Prediction and Evaluation...")
    
    print("   Running batch prediction on dataset...")
    start_time = time.time()
    
    # Get predictions for all texts
    all_predictions = []
    all_confidences = []
    
    for text in all_texts:
        result = classifier.predict_sentiment(text)
        all_predictions.append(result['sentiment'])
        all_confidences.append(result['confidence'])
    
    prediction_time = time.time() - start_time
    
    # Add results to dataframe
    df['predicted_sentiment'] = all_predictions
    df['confidence'] = all_confidences
    
    print(f"   ✅ Batch prediction completed! ({prediction_time:.2f}s)")
    
    # Calculate accuracy if ground truth labels exist
    if 'label' in df.columns:
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(df['label'], df['predicted_sentiment'])
        print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show classification report
        print("   📋 Classification Report:")
        report = classification_report(df['label'], df['predicted_sentiment'])
        print("   " + report.replace('\n', '\n   '))
    
    # 7. Save results
    print("\n7. 💾 Saving Results...")
    
    # Save predictions
    results_path = 'data/prediction_results.csv'
    df.to_csv(results_path, index=False)
    print(f"   ✅ Predictions saved to: {results_path}")
    
    # Save model and FAISS index
    model_path = 'models/trained_sentiment_model'
    os.makedirs(model_path, exist_ok=True)
    classifier.save_model(model_path)
    print(f"   ✅ Model and FAISS index saved to: {model_path}")
    
    # Create summary report
    summary = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "classification_model": classifier.classification_model_name,
            "embedding_model": classifier.embedding_model_name
        },
        "dataset_info": {
            "total_samples": len(df),
            "label_distribution": df['label'].value_counts().to_dict() if 'label' in df.columns else None
        },
        "performance": {
            "model_load_time": load_time,
            "embedding_creation_time": embedding_time,
            "batch_prediction_time": prediction_time,
            "accuracy": accuracy if 'label' in df.columns else None,
            "average_confidence": np.mean(all_confidences)
        },
        "technical_specs": {
            "embedding_dimensions": int(embeddings.shape[1]),
            "faiss_index_size": len(classifier.texts_database),
            "total_embeddings": int(embeddings.shape[0])
        }
    }
    
    summary_path = 'results_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✅ Summary report saved to: {summary_path}")
    
    # 8. Final summary
    print("\n8. 🎉 Pipeline Execution Complete!")
    print("=" * 60)
    print("📊 SUMMARY STATISTICS:")
    print(f"   • Total texts processed: {len(all_texts)}")
    print(f"   • Embedding dimensions: {embeddings.shape[1]}")
    print(f"   • Average confidence: {np.mean(all_confidences):.4f}")
    if 'label' in df.columns:
        print(f"   • Classification accuracy: {accuracy:.4f}")
    print(f"   • Total execution time: {time.time() - start_time + load_time:.2f}s")
    
    print("\n🚀 Next Steps:")
    print("   1. Run the Streamlit app: streamlit run app.py")
    print("   2. Explore the Jupyter notebook: notebooks/sentiment_analysis_workflow.ipynb")
    print("   3. Check the detailed results in: data/prediction_results.csv")
    
    print("\n✨ All requirements fulfilled:")
    print("   ✅ Hugging Face models (Transformers + sentence-transformers)")
    print("   ✅ Embeddings with FAISS similarity search")
    print("   ✅ Classification with example outputs")
    print("   ✅ Runnable script (.py)")
    print("   ✅ Sample dataset included")
    
    return classifier, df, summary


if __name__ == "__main__":
    classifier, results_df, summary = main()