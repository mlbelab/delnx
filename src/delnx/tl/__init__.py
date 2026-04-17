from ._de import de
from ._effects import log2fc
from ._glm_gp import GLMGPResult, NBFitResult, glm_gp, glm_gp_test, nb_de, nb_fit, nb_test
from ._grouped import grouped
from ._gsea import de_enrichment_analysis, single_enrichment_analysis
from ._rank_de import rank_de

__all__ = [
    "de",
    "grouped",
    "log2fc",
    "single_enrichment_analysis",
    "de_enrichment_analysis",
    "rank_de",
    "nb_fit",
    "nb_test",
    "nb_de",
    "NBFitResult",
    # Deprecated aliases
    "glm_gp",
    "glm_gp_test",
    "GLMGPResult",
]
