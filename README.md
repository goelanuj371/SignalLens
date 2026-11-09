
# SignalLens
### Aspect-Based Customer Feedback Intelligence

SignalLens is an end-to-end NLP system that transforms unstructured customer reviews into structured, actionable insights.

Unlike traditional sentiment analysis systems that assign a single sentiment to an entire review, SignalLens identifies individual product and service aspects, understands the sentiment associated with each aspect, and discovers recurring complaint patterns through unsupervised learning.

## Key Features

- **Aspect Term Extraction (ATE):** Identifies product and service attributes using a fine-tuned DistilBERT token classification model.
- **Aspect-Based Sentiment Analysis (ABSA):** Classifies sentiment towards each extracted aspect using domain-specific DistilBERT classifiers.
- **Complaint Discovery:** Uses sentence embeddings and unsupervised clustering to identify recurring negative feedback patterns.
- **Interactive Dashboard:** Explores sentiment distributions, problematic aspects, discovered complaint topics, and live review predictions.

## System Architecture

```text
             Customer Reviews
                    |
                    v
          Aspect Term Extraction
             (DistilBERT)
                    |
                    v
           Extracted Aspects
                    |
                    v
       Aspect-Based Sentiment Analysis
         (Domain-specific DistilBERT)
                    |
                    v
          Aspect-level Predictions
                    |
                    v
          Negative Feedback Records
                    |
                    v
          Sentence Embeddings
          (all-MiniLM-L6-v2)
                    |
                    v
             UMAP Reduction
                    |
                    v
             HDBSCAN Clustering
                    |
                    v
          Complaint Topic Discovery
                    |
                    v
           Interactive Dashboard
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| NLP Models | Hugging Face Transformers |
| Aspect Extraction | DistilBERT |
| Sentiment Classification | Domain-specific DistilBERT |
| Sentence Embeddings | Sentence Transformers |
| Dimensionality Reduction | UMAP |
| Clustering | HDBSCAN |
| Keyword Extraction | YAKE |
| Data Processing | Pandas, NumPy |
| Dashboard | Streamlit |

## Dataset

SignalLens uses the SemEval-2014 Task 4 Aspect-Based Sentiment Analysis dataset.

Two domains are supported:

- Laptop reviews
- Restaurant reviews

The dataset contains aspect-level annotations with four sentiment classes:

- Positive
- Negative
- Neutral
- Conflict

## Model Evaluation

The supervised models were evaluated using held-out test data.

| Model | Domain | Evaluation Metric | Score |
|---|---|---|---:|
| Aspect Extraction | Combined | Span F1 | 0.8219 |
| Aspect Sentiment | Laptop | Macro F1 | 0.6379 |
| Aspect Sentiment | Restaurant | Macro F1 | 0.6157 |

The integrated ATE + ABSA pipeline achieved a strict end-to-end F1 of 0.6448 on the evaluated test data.

These results reflect the evaluated SemEval domains and should not be interpreted as universal performance across arbitrary review categories.

## Complaint Discovery

Negative aspect-level feedback is converted into 384-dimensional sentence embeddings using `all-MiniLM-L6-v2`.

The embeddings are processed through:

1. UMAP for dimensionality reduction.
2. HDBSCAN for density-based clustering.
3. YAKE for keyword extraction.
4. Aspect recurrence analysis for topic descriptors.

The clustering pipeline discovers complaint themes without requiring predefined complaint categories.

Topic quality indicators are heuristic signals intended to support human inspection, not definitive measures of semantic correctness.

## Dashboard

The Streamlit dashboard provides four main sections:

1. **Overview:** Review statistics, sentiment distribution, problematic aspects and complaint topics.
2. **Aspect Intelligence:** Explore aspects with high negative sentiment.
3. **Complaint Discovery:** Inspect discovered clusters, keywords, quality indicators and representative reviews.
4. **Review Explorer:** Submit a new review for live aspect extraction and sentiment analysis.

## Project Structure

```text
SignalLens/
├── dashboard/
│   └── app.py
├── models/
│   ├── distilbert_ate/
│   ├── distilbert_laptop/
│   └── distilbert_restaurant/
├── notebooks/
│   └── SignalLens_ML_Pipeline.ipynb
├── outputs/
│   ├── aspect_level_predictions.csv
│   ├── aspect_sentiment_summary.csv
│   ├── complaint_topic_assignments.csv
│   ├── complaint_topic_summary.csv
│   ├── final_topic_report.csv
│   ├── negative_complaint_records.csv
│   ├── problematic_aspects.csv
│   ├── review_level_predictions.csv
│   └── topic_quality_report.csv
├── src/
│   └── signallens/
│       ├── models/
│       │   ├── ate.py
│       │   └── absa.py
│       └── pipeline.py
├── requirements.txt
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd SignalLens
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the CPU-compatible PyTorch build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

### 4. Add trained models

Download and extract the trained model archive. Place the three model directories inside `models/`:

- `distilbert_ate`
- `distilbert_laptop`
- `distilbert_restaurant`

The dashboard also requires the precomputed CSV artifacts inside `outputs/`.

### 5. Launch the dashboard

From the project root:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard will be available locally at:

```text
http://localhost:8501
```

## Limitations

- Supervised aspect extraction and sentiment models were trained and evaluated on laptop and restaurant review domains.
- Performance on unseen domains is not established.
- Conflict sentiment has comparatively weaker classification performance.
- Unsupervised complaint topics may contain noise or low-evidence clusters.
- Topic quality flags are heuristic and require human interpretation.
- The system's predictions should be treated as model-generated insights rather than verified ground truth.

## Future Improvements

- Domain adaptation for additional product categories.
- Improved handling of implicit and multi-word aspects.
- More robust conflict sentiment classification.
- Human-in-the-loop topic validation.
- Incremental complaint discovery as new reviews arrive.

---

**SignalLens — Turning customer feedback into structured intelligence.**
