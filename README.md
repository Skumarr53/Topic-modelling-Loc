# Centralized NLP Package

A centralized, modular, and scalable Python package for managing multiple NLP pipelines on Databricks. This package consolidates common functionalities for distribution to various stakeholders.

## Features

- **Data Access:** Seamlessly interact with Snowflake databases.
- **Preprocessing:** Efficient text tokenization, lemmatization, and n-gram processing.
- **Embedding Generation:** Train and manage Word2Vec models with optional bigram support.
- **Visualization:** Create interactive UMAP visualizations for embeddings.
- **Scalability:** Utilize Dask for parallel processing and handle large-scale data.
- **Configuration Management:** Flexible and modular configurations using Hydra.
- **Logging:** Structured logging with Loguru for easy debugging and monitoring.

## Installation

You can install the package using `pip`:

```bash
pip install centralized_nlp_package
