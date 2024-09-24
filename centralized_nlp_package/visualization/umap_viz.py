# centralized_nlp_package/visualization/umap_viz.py

import umap
import plotly.express as px
import pandas as pd
from typing import Optional
import numpy as np
from loguru import logger

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
