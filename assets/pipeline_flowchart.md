# Sentiment Analysis Pipeline Flowchart

## End-to-End System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Text    │───▶│ Text Preprocessing│───▶│  Tokenization   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                               ┌─────────────────────────┐│
                               │                         ▼
┌─────────────────┐           │    ┌─────────────────────────┐
│ Similarity      │◀──────────┤    │ Hugging Face Models     │
│ Search Results  │           │    │ ┌─────────────────────┐ │
└─────────────────┘           │    │ │  RoBERTa Classifier │ │
        ▲                     │    │ │  (Sentiment)        │ │
        │                     │    │ └─────────────────────┘ │
        │                     │    │ ┌─────────────────────┐ │
┌─────────────────┐           │    │ │  Sentence Trans.    │ │
│ FAISS Vector    │           │    │ │  (Embeddings)       │ │
│ Search Engine   │           │    │ └─────────────────────┘ │
└─────────────────┘           │    └─────────────────────────┘
        ▲                     │              │            │
        │                     │              ▼            ▼
┌─────────────────┐           │    ┌─────────────────┐  ┌──────────────┐
│ Embeddings      │           │    │ Sentiment       │  │ 384D Vector  │
│ Database        │◀──────────┤    │ Classification  │  │ Embeddings   │
└─────────────────┘           │    └─────────────────┘  └──────────────┘
                               │              │                  │
                               │              ▼                  │
                               │    ┌─────────────────────────────┤
                               │    │ Combined Results            │
                               │    │ • Sentiment + Confidence    │
                               └────│ • Similar Text Matches      │
                                    │ • Visualization Data        │
                                    └─────────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │ Output Interfaces           │
                               │ ┌─────────────────────────┐ │
                               │ │  Streamlit Web App      │ │
                               │ │  • Interactive UI       │ │
                               │ │  • Real-time Analysis   │ │
                               │ │  • Batch Processing     │ │
                               │ └─────────────────────────┘ │
                               │ ┌─────────────────────────┐ │
                               │ │  Python API             │ │
                               │ │  • Script Integration   │ │
                               │ │  • JSON Responses       │ │
                               │ └─────────────────────────┘ │
                               │ ┌─────────────────────────┐ │
                               │ │  Jupyter Notebook       │ │
                               │ │  • Analysis Workflow    │ │
                               │ │  • Visualizations       │ │
                               │ └─────────────────────────┘ │
                               └─────────────────────────────┘
```

## Component Details

### 1. Input Processing
- **Text Preprocessing**: Clean and normalize input text
- **Tokenization**: Convert text to model-compatible tokens
- **Batch Handling**: Process single or multiple texts

### 2. Hugging Face Models
- **Classification Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Input: Tokenized text
  - Output: Sentiment probabilities (positive/negative/neutral)
  - Confidence: Softmax scores for prediction confidence

- **Embedding Model**: `all-MiniLM-L6-v2`
  - Input: Raw text
  - Output: 384-dimensional dense vectors
  - Purpose: Semantic representation for similarity search

### 3. Vector Search System
- **FAISS Index**: Facebook AI Similarity Search
  - Index Type: IndexFlatIP (Inner Product)
  - Normalization: L2 normalized vectors for cosine similarity
  - Performance: Sub-millisecond search on large datasets

### 4. Output Integration
- **Sentiment Analysis**: Classification results with confidence scores
- **Similarity Search**: Ranked list of semantically similar texts
- **Combined Results**: Unified response with both sentiment and similarity data

### 5. User Interfaces
- **Streamlit App**: Interactive web application with modern UI
- **Python API**: Programmatic access via SentimentClassifier class
- **Jupyter Notebook**: Educational workflow with visualizations

## Data Flow

1. **Input** → Text enters the system
2. **Preprocessing** → Text cleaning and normalization
3. **Dual Processing**:
   - Path A: Classification pipeline for sentiment analysis
   - Path B: Embedding pipeline for vector representation
4. **FAISS Search** → Find similar texts using embeddings
5. **Results Combination** → Merge sentiment and similarity results
6. **Output** → Display through chosen interface (web, API, notebook)

## Key Technologies

- **Transformers**: Hugging Face model hub integration
- **PyTorch**: Deep learning framework for model inference
- **FAISS**: Efficient similarity search and clustering
- **Streamlit**: Rapid web application development
- **Plotly**: Interactive data visualizations
- **Pandas**: Data manipulation and analysis

## Performance Characteristics

- **Latency**: <100ms for single text analysis
- **Throughput**: 100+ texts per second (batch processing)
- **Memory**: ~500MB model footprint + dataset vectors
- **Scalability**: Handles 100K+ text similarity search efficiently