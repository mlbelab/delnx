from ._de import de
from ._design import build_design
from ._effects import auroc, log2fc
from ._glm_gp import GLMGPResult, NBFitResult, glm_gp, glm_gp_test, nb_de, nb_fit, nb_test
from ._grouped import grouped
from ._gsea import single_enrichment_analysis
from ._rank_de import rank_de

__all__ = [
    "de",
    "grouped",
    "build_design",
    "log2fc",
    "auroc",
    "single_enrichment_analysis",
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
