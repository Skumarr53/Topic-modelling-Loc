
import umap
import plotly.express as px
import pandas as pd
from typing import Optional
import numpy as np
from loguru import logger
from gensim.models import Word2Vec
import numpy as np
from loguru import logger
from centralized_nlp_package.preprocessing.text_preprocessing import tokenize_text
from centralized_nlp_package.preprocessing.ngram_utils import find_ngrams
from centralized_nlp_package.embedding.embedding_utils import average_token_embeddings
from centralized_nlp_package.preprocessing.ngram_utils import find_ngrams

def generate_ngram_embedding(x: str, model: Word2Vec) -> np.ndarray:
    """
    Computes the embedding for a given string `x` using a word2vec model. If bigrams exist in the model, 
    they are prioritized over unigrams.

    Args:
        x (str): The input string.
        model: The word2vec model that provides the word vectors.

    Returns:
        np.ndarray: The average embedding vector for the input string.
                    Returns None if no embedding can be found.
    """
    # Handle direct look-up for phrases with underscores
    if '_' in x:
        try:
            return model.wv[x]
        except KeyError:
            pass  # Continue to unigram/bigram handling if not found

    # Tokenize the input string into unigrams
    unigrams = tokenize_text(x)
    
    # Create a set of bigrams from the unigrams
    bigrams = [f"{b[0]}_{b[1]}" for b in generate_ngrams(unigrams, 2)]
    
    # Process bigrams and adjust unigrams list if bigrams are found in the model
    final_tokens = []
    prev_bigram_used = False
    
    for i, bigram in enumerate(bigrams):
        if bigram in model.wv:
            # If bigram exists in the model, use it and remove the corresponding unigrams
            final_tokens.append(bigram)
            if i == 0:  # Remove the first two unigrams for the first bigram
                unigrams.pop(0)
                unigrams.pop(0)
            else:  # For subsequent bigrams, just remove the second word
                unigrams.pop(1)
            prev_bigram_used = True
        else:
            prev_bigram_used = False
    
    # Add remaining unigrams that were not removed by bigrams
    final_tokens.extend(unigrams)
    
    # Compute the mean of the embeddings for the final tokens
    try:
        return average_token_embeddings(final_tokens,model) # np.mean(np.stack([model.wv[token] for token in final_tokens if token in model.wv]), axis=0)
    except ValueError:  # Catch empty stack
        try:
            return model.wv[x]  # Fallback to direct lookup of the original string
        except KeyError:
            return None

def find_nearest_words_with_embeddings(words: pd.DataFrame, model: Word2Vec, num_neigh: int = 50, regularize: bool = False) -> pd.DataFrame:
    """
    Finds the nearest neighbor words for each topic based on the embeddings.

    Args:
        words (pd.DataFrame): DataFrame containing 'label' and 'match' columns.
        model (Word2Vec): Trained Word2Vec model.
        num_neigh (int, optional): Number of neighbors to retrieve. Defaults to 50.
        regularize (bool, optional): Whether to apply cosine similarity normalization. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing labels, embeddings, matched words, and similarity scores.
    """
    logger.info("Finding nearest neighbors for each topic.")
    alist = {'label': [], 'embed': [], 'match': [], 'sim': []}
    for topic in set(words['label']):
        topic_embed = [[word[0], model.wv[word[0]], word[1]] for word in model.wv.most_similar_cosmul(
            positive=words[words['label'] == topic]['match'].apply(lambda x: [y for y in get_model_ngrams(x, model) if y in model.wv] if x not in model.wv else [x]).sum(),
            topn=num_neigh
        )]
        topic_embed_norm = [[word[0], model.wv[word[0]], word[1]] for word in model.wv.most_similar(
            positive=words[words['label'] == topic]['match'].apply(lambda x: [y for y in get_model_ngrams(x, model) if y in model.wv] if x not in model.wv else [x]).sum(),
            topn=num_neigh
        )]

        alist['label'] += [topic] * num_neigh
        if regularize:
            alist['embed'] += [embed[1] for embed in topic_embed]
            alist['match'] += [word[0] for word in topic_embed]
            alist['sim'] += [word[2] for word in topic_embed]
        else:
            alist['embed'] += [embed[1] for embed in topic_embed_norm]
            alist['match'] += [word[0] for word in topic_embed_norm]
            alist['sim'] += [word[2] for word in topic_embed_norm]

    tdf = pd.DataFrame(alist)
    if filename:
        # Save to JSON
        tdf.to_json(f"{filename}_neighbors_n{num_neigh}.json", orient="records")
    return tdf


def process_ngrams_tokens(x, model):
    """
    Tokenizes the input text and processes bigrams that exist in the model's vocabulary.

    Args:
        x (str): Input text.
        model: Word2Vec model with vocabulary 'wv'.

    Returns:
        list: Processed list of unigrams and bigrams.
    """
    unigrams = tokenize_text(x)
    bigrams = list(generate_ngrams(unigrams, 2))
    prev_removed = False

    if bigrams:
        bigram_joined = '_'.join(bigrams[0])
        if bigram_joined in model.wv:
            unigrams.remove(bigrams[0][0])
            unigrams.remove(bigrams[0][1])
            unigrams.append(bigram_joined)
            prev_removed = True

    for bigram in bigrams[1:]:
        bigram_joined = '_'.join(bigram)
        if bigram_joined in model.wv:
            unigrams.remove(bigram[1])
            unigrams.append(bigram_joined)
            if not prev_removed:
                unigrams.remove(bigram[0])
                prev_removed = True
        else:
            prev_removed = False

    return unigrams

# topic modelling
def compute_text_embedding(x, model):
    """
    Computes the embedding of the input text by averaging the embeddings of its unigrams and bigrams.

    Args:
        x (str): Input text.
        model: Word2Vec model with vocabulary 'wv'.

    Returns:
        numpy.ndarray or None: The embedding vector, or None if not found.
    """
    if '_' in x:
        try:
            return model.wv[x]
        except KeyError:
            pass  # Continue processing if the word is not in the vocabulary

    unigrams = process_ngrams_tokens(x, model)
    embeddings = [model.wv[phrase] for phrase in unigrams if phrase in model.wv]

    if embeddings:
        return np.mean(np.stack(embeddings), axis=0)
    else:
        try:
            return model.wv[x]
        except KeyError:
            return None
        

def umap_viz(df: pd.DataFrame, marker_size: Optional[int] = None, save_to: Optional[str] = None) -> None:
    """
    Generates a UMAP visualization for the provided embeddings.

    Args:
        df (pd.DataFrame): DataFrame containing 'embed' and 'label' columns.
        marker_size (Optional[int], optional): Size of the markers in the plot. Defaults to None.
        save_to (Optional[str], optional): Path to save the HTML visualization. Defaults to None.
    """
    logger.info("Generating UMAP visualization.")
    mapper = umap.UMAP().fit_transform(np.stack(df['embed']))
    df['x'] = mapper[:, 0]
    df['y'] = mapper[:, 1]
    fig = px.scatter(df, x='x', y='y', color='label', hover_data=['match'])
    fig.update_layout(
        autosize=False,
        width=1000,
        height=800,
    )
    if marker_size:
        fig.update_traces(marker_size=marker_size)
    if save_to:
        fig.write_html(save_to)
        logger.info(f"UMAP visualization saved to {save_to}")
    fig.show()
    logger.info("UMAP visualization generated successfully.")