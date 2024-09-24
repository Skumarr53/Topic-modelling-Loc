Usage: Pipeline 2
=================

This tutorial demonstrates how to run Pipeline 2: Embedding Generation and Visualization.

.. code-block:: bash

    python -m centralized_nlp_package.pipelines.input_preparation_pipeline2

.. code-block:: python

    from centralized_nlp_package.pipelines.input_preparation_pipeline2 import run_pipeline2
    from centralized_nlp_package.utils.config import get_config

    config = get_config()
    run_pipeline2(config)
