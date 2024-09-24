Usage: Pipeline 1
=================

This tutorial demonstrates how to run Pipeline 1: Model Inputs Preparation.

.. code-block:: bash

    python -m centralized_nlp_package.pipelines.input_preparation_pipeline1

.. code-block:: python

    from centralized_nlp_package.pipelines.input_preparation_pipeline1 import run_pipeline1
    from centralized_nlp_package.utils.config import get_config

    config = get_config()
    run_pipeline1(config)
