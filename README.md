# Movie Recommender RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for movie recommendations. This project combines a retriever (FAISS-based) and a generator (LLM-backed) to provide natural-language movie suggestions based on user queries. Supports streaming generation, conversation memory, and flexible retrieval criteria.

# Movie Recommender RAG Pipeline

A **Retrieval-Augmented Generation (RAG)** system for movie recommendations. This project combines a **retriever** (FAISS-based) and a **generator** (LLM-backed) to provide **natural-language movie suggestions** based on user queries. Supports **streaming generation**, **conversation memory**, and flexible retrieval criteria.


## Features

* **End-to-end movie recommendation pipeline**
  Query movies using natural language and receive explanations along with recommendations.

* **FAISS-based retrieval**
  Efficiently fetches relevant movie metadata, plot summaries, or genre-specific context.

* **LLM-backed generation**
  Uses Hugging Face models (like `meta-llama/Llama-2` or `sshleifer/tiny-gpt2`) to generate recommendations in natural language.

* **Streaming support**
  Incrementally generate text for a live typing experience.

* **Conversation memory**
  Tracks multiple queries and responses in a single session for context-aware recommendations.

* **Flexible retrieval options**
  Retrieve by genre, year range, or rating thresholds.


## Installation

```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

> **Optional:** For full LLM GPU support, install `accelerate`:
>
> ```bash
> pip install accelerate
> ```


## Dataset

This project uses the **MovieLens dataset** for educational purposes. Movie metadata, ratings, and genres are leveraged to build retrieval indices and context for the RAG pipeline.

### Citation

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)



## Usage

### 1. Preprocess the data

``` bash
python scripts/run_preprocessing.py
```

### 2. Instantiate the RAG pipeline

```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline with default retriever and generator
rag = RAGPipeline(max_context_tokens=1000)
```

### 3. Run a query

```python
query = "I like action movies with a strong plot"
response = rag.process_query(query)
print(response)
```

### 4. Streaming mode

```python
stream = rag.process_query("Show me comedies", top_k=5, stream=True)
for token in stream:
    print(token, end="", flush=True)
```

### 5. Reset conversation

```python
rag.reset_conversation()
```


## Testing

* **Preprocessing tests** :

```bash
python tests/test_preprocessing.py
```

* **Retriever tests:**

```bash
python tests/test_retriever.py
```

* **Generator tests:**

```bash
python tests/test_retriever.py
```

* **Lightweight tests** (mocked retriever and generator):

```bash
python tests/test_rag_pipeline.py
```

## Acknowledgements

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [FAISS](https://github.com/facebookresearch/faiss) for fast similarity search
* [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
* Inspiration from RAG pipelines in modern LLM applications

