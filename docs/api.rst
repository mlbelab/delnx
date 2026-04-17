📖 API reference
---------------

Preprocessing
~~~~~~~~~~~~~

.. module:: delnx.pp
.. currentmodule:: delnx.pp
.. autosummary::
    :toctree: genapi

    pseudobulk
    size_factors
    dispersion


Differential Expression
~~~~~~~~~~~~~~~~~~~~~~

.. module:: delnx.tl
.. currentmodule:: delnx.tl
.. autosummary::
    :toctree: genapi

    nb_fit
    nb_test
    nb_de
    NBFitResult
    de
    rank_de
    grouped


Effect Sizes
~~~~~~~~~~~~

.. currentmodule:: delnx.tl
.. autosummary::
    :toctree: genapi

    log2fc
    auroc


Utilities
~~~~~~~~~

.. currentmodule:: delnx.tl
.. autosummary::
    :toctree: genapi

    build_design
    single_enrichment_analysis


Models
~~~~~~

.. module:: delnx.models
.. currentmodule:: delnx.models
.. autosummary::
    :toctree: genapi

    LinearRegression
    LogisticRegression
    NegativeBinomialRegression
    DispersionEstimator


Plotting
~~~~~~~~

.. module:: delnx.pl
.. currentmodule:: delnx.pl
.. autosummary::
    :toctree: genapi

    dotplot
    gsea_dotplot
    gsea_heatmap
    heatmap
    matrixplot
    violinplot
    volcanoplot
